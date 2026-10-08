# 60. Gemma-4 prefill — attention 양자화·변환을 호스트에서 NPU로 (작업서, 2026-10-08)

59 §3의 A(Q 보정 스칼라 398 ms)·B(f32↔u16 변환 260 ms)·D/E(DSP append 양자화 168 ms)를 **고치는 대신
없애는** 작업이다. 배경·측정은 `59_gemma4_prefill_slowdown_handoff.md`(§2가 분해, §5가 함정),
그 전은 `58_gemma4_prefill_handoff.md`, `57_gemma4_all_npu_task.md` §9.19–§9.22. 규칙은 §8.

- 브랜치 `claude/zealous-bell-a2pot9` (Seunghui98/nntrainer), 이 문서 시점 HEAD `6d880861`(코드 `6a31ba72`).
- 기기 Galaxy S25 Ultra `R3CY10WM83Y`, Hexagon V79. 기기 실행과 skel 빌드는 사용자가 한다(클라우드 세션은
  skel을 컴파일할 수 없다). 숫자는 기기에서 잰 것만 숫자다. 나머지는 "기기 미측정".

---

## 0. 왜 이 방식인가 (QNN과의 비교)

QNN HTP 백엔드는 Quantize/Dequantize/Convert를 **그래프 안의 HTP op(HVX)**으로 둔다. 인코딩(scale,
offset)은 오프라인 보정으로 모델에 박혀 있고, 이웃 op끼리 형식이 맞아 activation이 NPU를 떠나지 않는다.
실행 중 호스트가 min/max를 재거나 형식을 바꾸는 일은 없다.

이 브랜치의 int8 attention은 그 반대다. `mha_core`가 f32 레이어라 PR 4343이 **호스트에서** Q 범위를 재고
f32↔u16을 바꾸며(PR 주석: "quantized graph에서는 이웃 레이어 형식이 이미 u16"이라는 전제의 임시 변환),
K/V는 fp16으로 DSP에 보내 DSP가 HMX 스레드 혼자 int8로 다시 양자화한다. 59 §2.3의 +0.3 s와 append 6.2 ms/층은
전부 이 "f32 레이어의 가장자리"에서 나온다. 01의 사다리 1단("존재해야 하나?")의 답이 "아니오"인 코드다.

목표: **qkv 호출이 Q를 u16로, K/V를 int8 마스터로 바로 쓰고, attention 호출은 f32 context를 돌려준다.**
호스트에는 보정도 변환도 남지 않는다.

---

## 1. 지금의 데이터 흐름 (as-built, sliding 층 1024행, 모두 59 §2 기기 실측)

| 순서 | 어디 | 무엇 | 비용 |
|---|---|---|---|
| 1 | qkv 호출 `QKVLayer::incremental_forwarding` → `HtpComputeOps::gemm_q4_0_batch_norm_fp32` → `invokeLayer` → skel `mm_u8i4_layer_norm` | f32 입력 11.5 MB를 ION으로 memcpy, DSP가 in_norm·u8 양자화·3 matmul·q/k/v head norm·RoPE, f32 Q 16 MB·K 8·V 8을 slice마다 행별 memcpy로 꺼냄(`htp_compute_ops.cpp:3300` 부근) | host= 16.1 ms + staging ≈2.8 ms |
| 2 | `MHACoreLayer::one_batch_incremental_forwarding` | K/V f32 → **호스트 fp16 KV cache**(`apply_rotary_emb_tensor_v2`, rope 없음) | kv_write 0.6 ms |
| 3 | `try_quantized_attention` 첫 호출 `calibrate_q2_scales`(`mha_core.cpp:1005`) | K head별 abs-max·V (head,dim)별 abs-max는 NEON, **Q head별 min/max는 스칼라** | 11.5 ms (full 22) |
| 4 | 같은 함수 `mha_core.cpp:1170` 부근 | **Q f32→u16 스칼라 단일 스레드**(`+0.5f` truncate, clamp) | ≈4 ms |
| 5 | `HtpComputeOps::sdpa_q2_kvcache`(`htp_compute_ops.cpp:6610`) | u16 Q 8 MB를 `attn_q_pool_` ION으로 memcpy, skel `attn_q2_step`: K/V fp16 행 8 MB 전송 → `hexkl_kv_q_append`(`append_fixed_i8`, HMX 스레드 혼자, 행·head마다 함수 호출 + `xor80_copy` 2차 패스) → `hexkl_attn_q2_prefill` | 전송 1.6 · append quant 6.2 + bake 1.0 · kernel 5.1 |
| 6 | 같은 함수 | u16 out 8 MB memcpy, `mha_core.cpp:1193` **u16→f32 스칼라** | ≈4 ms |
| 7 | o-proj 호출 | f32 context 16 MB를 다시 ION으로 memcpy | staging |
| decode | `htpDecodeAttention` → resident hook | 첫 decode에서 `decode_kv_seed_fp32`가 **호스트 fp16 KV cache**(위 2)에서 DSP를 seed(`mha_core.cpp:676-700`) | — |

30층 합: 보정 398 + 변환 260 + append quant 168 + kv_write 18 + K/V 전송·Q/out memcpy ≈ 50 → **≈0.9 s**.

---

## 2. 목표 데이터 흐름

| 순서 | 어디 | 무엇 |
|---|---|---|
| 1 | qkv 호출(새 변형) | norm·mm·post norm·RoPE 뒤 DSP에서: (a) Q head별 min/max(HVX, worker 분산) → `q_enc` 2×n_head_q, Q → **u16을 `attn_q_pool_` ION 버퍼에 직접**; (b) 첫 prefill이면 K head별 abs-max, V (head,dim)별 abs-max → `hexkl_kv_q_set_fixed_scales`; (c) K/V f32 행에서 **바로 int8 마스터 + 타일**(`hexkl_kv_q_append`의 f32 입력 변형, worker 분산); (d) K/V f32는 decode seed용으로 호스트에도 돌려줌(§4 3단계) |
| 2 | mha_core | 보정·변환 없음. `sdpa_q2_kvcache`를 "Q는 qkv가 남긴 버퍼, append 없음, out f32"로 호출 |
| 3 | attention 호출 `attn_q2_step` | `k_rows`/`v_rows` 빈 시퀀스, Q는 ION 버퍼, **out f32**(`stage_epilogue`가 u16으로 묶기 전 f32 값을 이미 가진다, `hexkl_attn_q2.c:317`) |
| 4 | o-proj | 변함없음 |

없어지는 것: 1·3·4·6의 호스트 루프 전부, 5의 append quant와 K/V 전송, Q/out memcpy 16 MB. 남는 것: kv_write
0.6 ms(decode seed, §4 4단계에서 선택 제거), bake 1.0 ms(qkv 호출로 이동), 전송·kernel.

기대(산술, 기기 미측정): prefill 4.79 → **≈3.9 s**. 59 §3 A+B(호스트 NEON)의 −0.6 s보다 크고 코드는 줄어든다.

---

## 3. 설계 결정 (바꾸려면 측정으로)

1. **범위 정의는 지금과 같게.** Q head별 min/max(u16 비대칭), K head별 abs-max, V (head,dim)별 abs-max,
   `finish`(max/127, 0이면 1). 호스트 보정과 정의가 같으면 nll ≈ 3.508이 그대로 나와야 한다(실행 노이즈
   안). 다르면 수정이 틀린 것이지 범위 정의를 바꿀 일이 아니다. 정적 인코딩(QNN의 PTQ 방식)은 prompt가
   바뀌면 Q·V 범위가 넘칠 수 있어 **채택하지 않는다**. 동적(호출마다 그 행에서) 보정이 HVX 두 패스로 층당
   1–2 ms라 충분히 싸다.
2. **반올림 정의는 DSP 것(`hvx_sf_to_w_rne`, RNE)으로 통일.** 지금 호스트 `+0.5f` truncate와 다르지만 호스트
   코드는 참조가 아니다(그리고 앱은 `-ffast-math`라 그 코드의 비트 동일성 자체가 정의되지 않는다, 59 §5).
   게이트는 nll과 문장이다. PR 4343의 기기 gtest가 Q를 호스트에서 양자화해 SNR을 보므로 그 참조를 RNE로 맞춘다.
3. **attention 출력은 f32.** `hexkl_attn_q2_io`에 `float *out_f32`를 더하고 u16 `out`은 PR 테스트용으로
   남긴다(둘 중 하나가 NULL). `out_enc`는 f32 경로에서 쓰지 않는다. IDL은 `attn_q2_step`에 `rout
   sequence<float> out_f32`를 더하거나 새 메서드로 — 어느 쪽이든 **skel과 앱 stub 둘 다 재생성**(58 §7 함정 1).
4. **Q u16 버퍼의 수명.** qkv 호출이 쓴 Q는 바로 다음 노드인 attention 호출까지 살아 있어야 한다. `act_pool_`/
   `out_pool_`은 다음 호출이 덮으므로 쓰면 안 되고, `attn_q_pool_`(attention 전용 풀)에 쓴다. 전달은
   `HtpComputeOps`가 "마지막 qkv 호출이 남긴 Q(버퍼, enc, 행 수, 층 식별)"를 들고 있고 `sdpa_q2_kvcache`가
   `q == nullptr`일 때 그것을 쓰는 형태가 가장 작다. 그래프 순서가 qkv → attention을 보장한다. 층 식별이
   어긋나면(다른 층의 Q) 실패로 처리하고 CPU 경로로 떨어진다(조용히 틀리지 않게). `ponytail:` 주석으로
   남기고, 상한은 "qkv와 attention 사이에 다른 HTP 호출이 끼는 그래프"라고 적는다.
5. **KV handle 공유.** 지금은 `mha_core`가 첫 attention 호출에서 `kv_cache_q_register`를 lazy로 한다
   (`q_cache_handles[batch]`). qkv 호출이 먼저 돌아 append를 하려면 handle이 먼저 있어야 한다. 선택지:
   (a) 등록을 `QKVLayer`로 옮기고 `mha_core`가 받아 쓴다, (b) `allocateAndBindKVCache` 때 층별로 등록해 두
   레이어가 층 번호로 찾는다. 두 레이어는 이름이 `layerN_qkv`/`layerN_attention`이고 같은 블록이다. 코드가
   적은 쪽으로, 단 batch>1(`q_cache_handles`가 batch별)과 `release_quantized_cache`·실패 시 fp16 경로
   fallback(`q_cache_failed`)은 유지한다.
6. **decode는 건드리지 않는다.** decode attention은 resident hook(`htpDecodeAttention`)이고 Q f32·호스트 fp16
   KV cache를 쓴다. 이 작업은 **M > 1(prefill)에서만** 켠다. M == 1에서 `gemm_q4_0_batch_norm_fp32`가 HTP를
   타는 경우(`accelerates_qs4cx_at_m1`) Q u16 출력·K/V append는 꺼져 있어야 한다.
7. **호스트 fp16 KV cache 쓰기(kv_write 0.6 ms/층)는 유지**한다(decode seed가 읽는다). 없애는 것은 4단계(선택).
8. **Q f32 출력 복사(16 MB/층) 생략은 측정 뒤.** 첫 버전은 Q f32도 그대로 꺼내 둔다(q2 경로가 실패하면 CPU
   attention이 그 Q를 읽는다). u16 Q만으로 충분하다고 기기에서 확인되면 생략한다(≈1 ms/층).
9. **누산은 IEEE sf, qf32 금지**(59 §5의 router 사고). min/max·abs-max는 `Q6_Vsf_vmax/vmin`으로 충분하다.
10. HVX 코드는 **기기에서 CPU와 행 단위 비교**가 통과해야 "됐다"다. 호스트 lane 에뮬 통과는 컴파일 검사일 뿐.

---

## 4. 단계 (각 단계 = 한 주제 한 커밋, 분해 → 수정 → 정확도 게이트 → 속도, 결과는 57 §9 새 절)

### 1단계: attention 출력 f32 (B의 절반 + out memcpy)

- DSP: `hexkl_attn_q2.c` `stage_epilogue` — `out_f32`가 있으면 `f0/f1`(dequant 전 int → f32에 `c0/c1` 곱)을
  `(acc * s_v[d] * 512/2^kv / 65535)`로 바로 f32 저장. 지금 식 `out = sat_u16(round(o16*c[d] + zp_o))`에서
  u16 인코딩을 벗긴 값이다(`c[d] = 512/2^kv * s_v[d] / (65535 * s_o)`, 즉 f32 값 = `o16 * 512/2^kv * s_v[d] /
  65535`). 비트 동일성 요구 없음, nll로.
- skel `nntr_hvx_attn_q.c` `attn_q2_step`: `out_f32` 시퀀스(길이 0이면 u16 경로).
- IDL `test/htp/nntr_hvx.idl`: `attn_q2_step`에 `rout sequence<float> out_f32` 추가(기존 `out_u16`은 길이 0 허용).
- host `htp_compute_ops.cpp` `sdpa_q2_kvcache`: `out` f32 포인터(ION `attn_out_pool_` f32 → 호출자 버퍼
  memcpy 16 MB; 또는 호출자 버퍼가 ION이면 직접), `mha_core.cpp`의 u16→f32 루프 삭제, `q2_out_u16`·`q2_out_enc`
  삭제.
- 검사: 기기 gtest(`test/unittest/unittest_hvx_attn_q.cpp`)에 f32 out 케이스 1개(SNR 같은 기준), 호스트
  `run_host_checks.sh`, `unittest_causallm_models`.
- 기대: −130 ms(변환) − 8 MB memcpy ≈ −150 ms(기기 미측정).

### 2단계: Q 양자화를 qkv 호출로 (A의 Q 몫 + B의 나머지 + Q memcpy)

- IDL: `mm_u8i4_layer_norm`의 변형(새 메서드 권장, 기존 호출자 불변): `in uint32 q16_handle`(어느 handle의
  출력이 Q인지, 보통 0), `rout sequence<uint16> q_u16`, `rout sequence<float> q_enc`(2×n_head_q), `in uint32
  q16_head_dim`. DSP: 그 handle의 `out_cat` 블록(M×N_q f32, 이미 post norm·RoPE 끝)에 대해 head별 min/max
  (HVX, `quant_pool` worker에 행 분산, 스레드별 결과를 HMX 스레드가 병합) → `q_enc` → u16 RNE 변환을
  `q_u16`에. `q_u16`은 호스트가 `attn_q_pool_` ION 버퍼를 넘긴 것이다.
- host: `HtpComputeOps::gemm_q4_0_batch_norm_fp32`에 "Q를 u16로도" 옵션(`QKVLayer`가 `attention_kv_dtype
  q8`·M>1·head_dim 조건일 때 켬 — 조건은 지금 `try_quantized_attention`의 `use_q2` 조건과 같아야 한다).
  결과(버퍼, enc, M, 층 식별)를 `HtpComputeOps`가 보관(§3 4). `sdpa_q2_kvcache`는 `q == nullptr`이면 그것을
  쓴다. `QKVLayer`가 `use_q2` 조건을 알려면 property 하나(`attn_q16`)를 모델 빌더가 `attention_kv_dtype`에서
  같이 넣는 것이 가장 작다(`gemma4_causallm.cpp` `createAttention`).
- `mha_core.cpp`: `calibrate_q2_scales`에서 Q 부분(`q_lo/q_hi`, `q2_q_enc`) 삭제, f32→u16 루프와 `q2_q_u16`
  삭제. `q2_calibrated`는 K/V에만 남는다(3단계에서 사라짐).
- 검사: 기기에서 첫 층의 `q_u16`·`q_enc`를 dump해 CPU 계산(같은 f32 Q에서 min/max·RNE)과 행 단위 비교
  (임시 계측, 커밋 안 함; 59 §5 `NNTR_MOE_DIFF` 방식). 호스트 에뮬에 min/max·u16 패스 검사 1개
  (`test/htp/host/`).
- 기대: −398 −130 − 8 MB memcpy ≈ −540 ms(기기 미측정).

### 3단계: K/V 양자화를 qkv 호출로 (append quant + K/V 전송)

- `hexkl_kv_q.c`: `hexkl_kv_q_append`의 **f32 행 입력 변형**(`hvx_kv_quant.c`의 `load_row_f32`는 fp16을
  넓히는 것이라 f32 입력은 그냥 벡터 로드; `pack_bytes`가 바로 u8(xor 0x80)을 쓰게 해 `xor80_copy` 패스
  제거) + 행을 `quant_pool` worker에 분산(E). 첫 prefill이면 같은 호출 안에서 K head별·V (head,dim)별 abs-max
  → `hexkl_kv_q_set_fixed_scales`(지금 `kv_cache_q_set_fixed_scales` 호출이 하던 것).
- IDL: 2단계 메서드에 `in uint32 kv_handle, in uint32 row0, in uint32 k_handle, in uint32 v_handle`
  (어느 handle 출력이 K/V인지), 첫 호출 플래그. 0이면 K/V 양자화 안 함.
- host: handle 공유(§3 5). `attn_q2_step`의 `k_rows`/`v_rows`는 빈 시퀀스. `mha_core`의 `calibrate_q2_scales`·
  `q2_scales_set`·`kv_cache_q_set_fixed_scales` 호출 삭제. 호스트 fp16 KV cache 쓰기는 유지(§3 7).
- 검사: 기기에서 첫 층 int8 마스터(`hexkl_kv_q_dump`가 있다)를 CPU 양자화와 행 비교. 기기 gtest
  `unittest_hexkl_kv_q.cpp`·`unittest_hexkl_kv_quant.cpp`에 f32 입력 케이스.
- 기대: −168(quant) − K/V 8 MB 전송 ≈ −200 ms(기기 미측정). bake 1.0 ms는 qkv 호출로 옮겨 갈 뿐 남는다.

### 4단계(선택): decode seed를 DSP 마스터에서

`decode_kv_seed_fp32`가 호스트 fp16 cache 대신 int8 마스터를 dequant해 seed하면 kv_write 18 ms와 fp16 KV
cache 메모리(seq×2048×2 B×2×30층)가 사라진다. 정확도가 바뀌므로(fp16 → int8 seed) 생성 16토큰 게이트로.
작아서 1–3단계 뒤 남는 몫을 보고 결정.

---

## 5. 게이트와 명령

빌드·push·정확도·속도 명령은 **59 §4 그대로**(skel은 `test/htp/build.sh`, IDL이 바뀌었으면 앱은 `--cache`
금지). 추가로 매 단계:

```bash
# 클라우드(host): 59/58 §6.1 그대로
ninja -C build && ./build/Applications/CausalLM/unittest_causallm_models && bash test/htp/host/run_host_checks.sh
# 사용자 PC: skel + 앱 + 기기 gtest(PR 4343의 unittest_hvx_attn_q, kv_q, kv_quant를 기기에 push해 실행)
# 정확도: 59 §4.2  (기대 nll ≈ 3.51, "…harbour town of Ardley…")
# 속도: 59 §4.3  (prefill, staging memcpy, attn trace; 냉각 2분, 2–3회)
```

트레이스는 새 단계가 보이게 넓힌다: `attn_q2_step` stats에 append 0이 찍히는지, qkv 호출에 Q/K/V 양자화
us(`NNTR_HTP_ATTN_TRACE`가 켜졌을 때 `mm_u8i4_layer_norm` 변형이 stats로 돌려주거나 FARF). 단계별 판정:

| 단계 | 정확도 | 속도(cfgB prefill, 기기 미측정 기대) | trace |
|---|---|---|---|
| 1 | nll ≈ 3.51 | 4.79 → ≈4.64 | host 나머지 7.8 → ≈4 ms/층 |
| 2 | nll ≈ 3.51 | → ≈4.1 | `q2 calibration us` 줄 사라짐, host 나머지 ≈0 |
| 3 | nll ≈ 3.51 | → ≈3.9 | `append= (quant=0 …)`, qkv 호출 +1–2 ms |

---

## 6. 함정 (59 §5·58 §7 + 이 작업 고유)

- IDL이 바뀌면 skel(`test/htp/generated/`)과 앱 stub(`nntrainer/tensor/htp_backend/generated/`) **둘 다**.
  하나만 옛것이면 `0x8000040e` 또는 엉뚱한 함수. `build.sh`의 `undefined symbols: ldexpf …` exit 1은 원래 있던
  것, `error:` 줄만 본다.
- `-ffast-math`(앱): NaN 검사는 비트로. "호스트 스칼라와 비트 동일" 주장은 이 앱 코드에 대해 성립하지 않는다.
- qf32 누산 금지, 호스트 lane 에뮬 ≠ 기기. 새 HVX 패스는 기기에서 행 비교.
- PR 4343 코드는 `hvx_worker_pool_submit`이 int 반환 기준, 이 브랜치는 void(6a31ba72). 가져올 때마다 같은 대응.
- `attn_q_pool_`은 size class 풀이다(`stage()`가 64 KiB부터 2배씩). full 층(Q 16 MB)과 sliding 층(8 MB)이
  다른 class를 쓰므로 "마지막 qkv의 Q" 기록에 버퍼 포인터를 함께 둔다.
- `skip_prefill: true`면 PPL이 안 찍힌다. 정확도는 `cfgB_ppl.json`.
- MoE를 CPU로 돌리면 기기 연결이 끊긴다. 엔진 사다리는 MoE htp 고정.
- 프로파일 빌드는 전체가 17% 느리다(59 §2 by-op 5,617 vs 4,791). 절대값 비교는 일반 빌드 `[HTP-PROFILE]`+trace로.

---

## 7. 코드 지도 (이 작업이 건드리는 곳)

| 영역 | 파일 |
|---|---|
| 호스트 attention 레이어 | `Applications/CausalLM/layers/mha_core.cpp` (`try_quantized_attention` 1056행~, `calibrate_q2_scales` 1005행~), `mha_core.h` |
| 호스트 qkv 레이어 | `Applications/CausalLM/layers/qkv_layer.cpp` (`incremental_forwarding` 325행~, HTP 분기 390행~) |
| 모델 그래프 | `Applications/CausalLM/models/gemma4/gemma4_causallm.cpp` `createAttention`(qkv_layer·attention core 파라미터) |
| HTP 호스트 쪽 | `nntrainer/tensor/htp_backend/htp_compute_ops.cpp` (`invokeLayer` 3255행~, `gemm_q4_0_batch_norm_fp32` 1300행, `sdpa_q2_kvcache` 6610행, `stage()` 3168행), `nntrainer/tensor/cpu_backend/compute_ops.h`(가상 함수 선언) |
| DSP 진입점 | `test/htp/nntr_hvx.idl` (`mm_u8i4_layer_norm` 115행, `attn_q2_step` 1167행), `test/htp/nntr_hvx_mm_u8i4.c` (`mm_u8i4_layer_norm` 800행 부근), `test/htp/nntr_hvx_attn_q.c` (`attn_q2_step` 434행) |
| DSP 커널 | `nntrainer/tensor/htp_backend/hmx/hexkl_attn_q2.c` (`stage_epilogue` 317행), `hexkl_attn_q2.h` (`hexkl_attn_q2_io`), `hexkl_kv_q.c` (`append_fixed_i8` 430행, `hexkl_kv_q_append` 596행, `hexkl_kv_q_set_fixed_scales` 129행), `hvx/hvx_kv_quant.c`, `hvx/hvx_worker_pool.h` |
| 테스트 | `test/unittest/unittest_hvx_attn_q.cpp`, `unittest_hexkl_kv_q.cpp`, `unittest_hexkl_kv_quant.cpp`, `unittest_cpu_kv_q_attention.cpp`, `test/htp/host/run_host_checks.sh` |
| 도구 | `tools/prefill_timeline.py`, `tools/htp_syntax_check.sh` |

---

## 8. 작업 규칙

58 §9·59 §7 그대로. `CLAUDE.md`, `AGENTS.md`, `01_working_style.md`(ponytail). 커밋 제목 `[component] subject`,
한 주제 한 커밋, author `SeungHui Lee <shsh1004.lee@samsung.com>`, committer Claude, trailer `Co-authored-by:
Claude <noreply@anthropic.com>` + `Signed-off-by: SeungHui Lee <shsh1004.lee@samsung.com>` + 세션 attribution 줄.
바뀐 줄만 clang-format. `subprojects/` 수정 금지. 결과는 57 §9 새 절, 기기에서 안 잰 숫자는 "기기 미측정".
각 단계는 측정 분해 → 수정 → 정확도 게이트 → 속도. 사용자에게는 한국어로, 기기 명령은 복붙 가능한 한 블록으로.
