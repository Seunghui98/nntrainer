# 47 — prefill 181 → 353 TPS: 무엇을 바꿨고, 각각이 왜 빨라졌나 (2026-09-15)

같은 기기, 같은 모델(`lfm2.5-8b-a1b-q40-qs4cx-wh`, QS4CX_WH 아레나), 같은 프롬프트
(444 토큰, 512 생성), 같은 명령(`NNTR_HTP_PROFILE=2`, `--profile` 빌드 아님, 스레드
기본값)으로 두 번 돌린 결과다. 각 실행은 **1회**이고 DSP 시간의 실행 간 변동은 ±4 ms
수준(문서 46 §23.2)이라, 0.2 ms 단위의 차이는 읽지 않는다. 텍스트는 두 실행 모두 정상
(같은 3문장 요약).

| | **before** (`e7ae879` 상태) | **after** (`164ca3d` + `fdf2c97` + `f48d109`) | 배수 |
|---|---:|---:|---:|
| **prefill** | 2449 ms, **181.3 TPS** | 1257 ms, **353.2 TPS** | **1.95×** |
| decode | 84.7 ms/token, 11.81 TPS | 67.3 ms/token, 14.87 TPS | 1.26× |
| MoE 레이어 콜 (prefill, 22콜) | 28.2 ms/콜, 620.7 ms | 22.6 ms/콜, 496.8 ms | 1.25× |
| MoE 레이어 콜 (decode, 22콜/토큰) | 2.71 ms/콜 | 2.02 ms/콜 | 1.34× |
| 피크 메모리 | 5330 MB | 5326 MB | — |

코드 변경은 셋이고, 셋이 각각 다른 곳을 줄였다. 아래는 하나씩, **before의 메커니즘 →
무엇을 바꿨나 → 프로파일의 어느 숫자가 움직였나** 순이다.

---

## 1. 등록을 로드 시점으로 (P6, `f48d109`) — prefill 벽시계 −747 ms

### before

HTP는 전문가 가중치를 **처음 쓰는 forward에서** 등록했다(`get_or_register_wh`, 포인터
캐시 미스 → 아레나 memcpy → FastRPC 등록 → `madvise`). 1408개 전부가 첫 prefill
안에서 일어났고, 프로파일이 그 시간을 따로 찍는다:

```
[HTP-PROFILE]   registration total :      746.8 ms
prefill: 444 tokens, 2449 ms
```

prefill 2449 = **등록 747** + 레이어 콜 621 + 나머지 ARM 1081. 등록이 prefill의 31%였다.
등록은 모델을 한 번 올릴 때 하는 일이지 토큰마다·프롬프트마다 하는 일이 아닌데,
"첫 forward에 지연 등록"이라는 구현 편의 때문에 prefill TPS 안에 들어가 있었다.

### 바꾼 것

`Transformer::repack_weight()`는 로드 직후(`main.cpp`: `load_weight` → `repack_weight`
→ `run`) 모든 레이어의 가중치를 한 번 순회하며 QS4CX를 `pack()`한다. 그 순회에
**한 줄을 더했다**: `lfm2_moe` 레이어이고 컨텍스트에 ComputeOps가 붙어 있으면(=
`nntr_config.json`이 그 레이어를 htp로 보냈으면) QS4CX/QS4CX_WH 가중치마다
`ComputeOps::register_qs4cx_weight(data, scale, K, N, wh)`를 부른다. HTP 백엔드는 이걸
forward 때와 **같은** `get_or_register_wh`로 답한다 — 같은 데이터 포인터가 캐시 키라
첫 forward의 등록은 전부 캐시 히트가 된다. 다른 백엔드는 `false`를 돌려주고 건너뛴다.

파일: `nntrainer/tensor/cpu_backend/compute_ops.h` (가상 함수 1개),
`nntrainer/tensor/htp_backend/htp_compute_ops.cpp` (구현 12줄),
`Applications/CausalLM/models/transformer.cpp` (순회 람다).

### after

```
[HTP-PROFILE]   registration total :      735.6 ms     ← 그대로 찍힌다. 로드 단계에서
prefill: 444 tokens, 1257 ms                            ← 벽시계에서 빠졌다
```

**주의해서 읽을 것:** 등록 비용 자체는 안 줄었다(747 → 736, 변동 범위). 사라진 게 아니라
**모델 로드 시간으로 옮겨간 것**이다. 이게 맞는 이유: (1) 등록은 프롬프트와 무관한
1회성 작업, (2) `--profile` 노드 표에서 HTP 레이어의 첫 콜이 등록으로 오염되던 것이
풀려 CPU 레이어와 바로 비교할 수 있게 됐다(문서 46 §50.4).

부수효과로 **나머지 ARM도 1081 → 760 ms** 줄었다. 등록의 memcpy·madvise가 첫 forward
안에서 다른 레이어의 페이지 폴트·캐시와 섞여 있었을 가능성이 크지만 **이 귀속은 재지
않았다.** 확인하려면 `--profile` 빌드로 두 실행의 레이어 TYPE 합계를 비교하면 된다.

---

## 2. DMA 순서 교정 (P2, `164ca3d`) — 콜당 −2.8 ms, decode 콜당 −0.8 ms

### before — "gather 고정비"의 정체

프로파일의 prefill 콜 한 줄:

```
gather 2965.3   drain 9.3+4.3
```

`gather`는 64행 활성화 블록(128 KB)을 힙 → VTCM으로 DMA하고 기다리는 시간이다. 128 KB가
2965 us일 리 없다. 그리고 `drain`(가중치 DMA 대기)이 13 us — 176 MB를 스트리밍하는
콜에서 대기가 0에 가까운 것도 말이 안 된다. 둘을 합쳐 읽으면 답이 나온다:

`hexkl_dma_ring.c:128`: *"Descriptors are dmlinked, so they retire in push order and
waiting on one implies every earlier one."* — 링은 **큐 순서로만 완료**되고, 어떤
디스크립터를 기다리는 것은 그 앞의 모든 디스크립터를 기다리는 것이다.

커널의 expert당 순서(before):

```
push down[e]        (2 청크, 1.75 MB)
push act 블록       (128 KB)
wait(act)                      ← GATHER로 계측. 앞에 줄 선 gate_up[e] 잔여 + down[e]를 다 기다린다
wait(gate_up 청크 0..3)        ← DRAIN으로 계측. 이미 끝나 있어 ≈0
```

그래서 (1) 가중치 대기가 `gather` 버킷에 찍혔고, (2) **더 나쁘게**, expert의 첫 블록은
gate_up 5.25 MB가 전부 도착할 때까지 HMX를 한 타일도 못 돌렸다. 문서 46 §26.4가
"청크 스트리밍으로 첫 청크(1 MB)만 오면 시작"이라고 만든 파이프라인이 첫 블록에서는
무효였다 — R2b(§29.3)가 활성화 복사를 DMA로 바꾸면서 그 대기가 DRAIN에서 GATHER로
**자리만 옮긴 것**이라 아무도 못 봤다.

decode는 expert마다 블록이 하나라 **모든** 블록이 첫 블록이다: `gather 809 us` — 문서 46
§47.2(a)가 "M과 무관한 769 us 고정비"라고 부른 것이 정확히 이것이다.

### 바꾼 것

활성화 블록을 그 expert의 가중치 push **앞에** 큐잉한다:

```
expert 0 블록 0    : 루프 전, gate_up[0] push 앞
expert e+1 블록 0  : expert e의 마지막 블록에서 gate_up mm이 끝난 직후 (act 슬롯이 죽는
                     시점), gate_up[e+1] prefetch 앞
down[e]            : 블록 0의 act wait 뒤로
블록 1 이상        : 제자리 (앞에 줄 선 건 자기 expert의 down뿐, 그건 이미 끝나 있다)
```

push 횟수는 같다(링 예산 불변). `moe_push_act_block()` 하나로 뽑아 세 곳에서 부른다.
호스트 체크의 DMA 스텁은 즉시 완료라 순서의 효과는 기기에서만 보였다.

### after

```
                 before      after
prefill gather   2965.3  →   125.6    (−2840)   ← 128 KB × 45.6 블록, 진짜 gather
prefill drain    9.3+4.3 →   225+58   (+269)    ← 가중치 대기가 제 버킷으로 갔고, 대부분 HMX 뒤에 숨는다
decode  gather   809.4   →   2.2      (−807)
decode  drain    0.9+0.5 →   189+8    (+196)    ← decode는 숨길 HMX가 짧아 일부 노출
decode  dsp      2060.6  →   1449.6   (−611)
```

**예측과 대조:** 문서 46 §49.2가 "gather 795 → ≤20, 아니면 진단이 틀린 것"이라고 적었다.
2.2가 나왔다. prefill은 "≈200" 예측에 126.

첫 청크 대기가 처음으로 진짜 전송을 재게 됐다: `first 1024 KB took 58 us = 18.1 GB/s`
(prefill), `64 us = 16.4 GB/s` (decode). before에는 `0 us = 23068 GB/s`라는 무의미한
값이었다. **부하 중 DMA는 16–18 GB/s이지 Gate 0c의 38.8이 아니다** — 이게 decode의
다음 방향을 정하는 숫자다(문서 46 §50.7).

---

## 3. 세션 스크래치 (P1, `fdf2c97` + `2bf893f`) — 콜당 −2.7 ms (−3.0 예정)

### before

커널은 콜마다 13개 버퍼를 `malloc`하고 끝에 `free`했다: 슬롯 순서 활성화(n_slots × K
≈ 6 MB), 활성화의 캐시된 사본(3.6 MB), 출력(3.6 MB), 테이블들. 합쳐 ~12.8 MB.
`HEXKL_PROBE_ALLOC`(§30.4 R3-2가 넣은 프로브)이 그 비용을 직접 읽었다:

```
alloc 3063.7 us   (prefill 콜당)
```

계산이 아니라 힙 관리 — 큰 블록의 malloc/free는 페이지 매핑·반납을 동반한다.

### 바꾼 것

`hexkl_moe_scratch` 한 블록을 세션(`nntr_hvx_session.moe_scratch`)이 들고, 커널은
필요한 합계를 계산해 모자랄 때만 키우고(`moe_scratch_reserve`) 13개 포인터를 128 B
정렬로 잘라 쓴다(`moe_carve`). 커지기만 하고 `close()`에서 푼다. decode는 prefill이
키워 둔 블록의 1/20만 쓴다.

### after

```
prefill alloc   3063.7  →   323.1     (−2741)
decode  alloc   9.5     →   0.2
```

323이 남은 이유: 레이어마다 라우팅이 달라 `n_slots`가 다르고, 첫 실행 안에서 더 넓게
라우팅된 레이어를 만날 때마다 블록이 다시 자랐다. `2bf893f`가 상한(`n_rows +
n_experts × 63` — 모든 expert가 한 블록씩 패딩된 최악)으로 예약하게 바꿔 첫 콜에 한 번만
자라게 했다. **이건 아직 기기에서 안 쟀다** — 다음 실행에서 `alloc ≈ 0`이어야 한다.

---

## 4. 단계별 before/after — prefill 콜 하나 (us)

| 단계 | before | after | Δ | 원인 |
|---|---:|---:|---:|---|
| gather | 2965.3 | 125.6 | **−2839.7** | §2 |
| alloc | 3063.7 | 323.1 | **−2740.6** | §3 |
| quant | 1855.0 | 1645.5 | −209.5 | DDR 경합 감소로 추정 (미검증) |
| scatter | 1274.0 | 1085.5 | −188.5 | 〃 |
| drain | 9.3+4.3 | 225.2+57.5 | +269.1 | §2 — 제 버킷으로 돌아온 가중치 대기 |
| mm | 8586.0 | 8778.9 | +192.9 | 변동 범위 (HMX 발행, 블록 수 동일 1004) |
| acc | 2980.5 | 2945.5 | −35.0 | — |
| dequant | 1378.5 | 1363.0 | −15.5 | — |
| requant | 1219.7 | 1186.1 | −33.6 | — |
| swiglu | 2050.1 | 2054.6 | +4.5 | — |
| stage | 389.5 | 381.9 | −7.6 | — |
| push / rest | 19.5 / 89.7 | 22.2 / 84.8 | — | — |
| **dsp** | **25885.2** | **20279.1** | **−5606.1** | |
| transport | 2327.6 | 2301.4 | −26.2 | 안 건드렸다 |
| **host** | **28212.8** | **22580.5** | **−5632.3** | |

두 변경(§2, §3)이 −5580을 설명하고 나머지 −26은 변동이다. **HMX(mm+acc 11.7 ms)는
그대로다** — 이 두 변경은 "낭비"를 없앤 것이지 계산을 빠르게 한 게 아니다.

## 5. 전체 prefill 벽시계 분해 (ms)

| | before | after | Δ |
|---|---:|---:|---:|
| 등록 (첫 forward 안) | 746.8 | 0 (로드로) | −746.8 |
| HTP 레이어 콜 22개 | 620.7 | 496.8 | −123.9 |
| 나머지 ARM (conv·attn·norm·router·staging) | 1081.5 | 760.2 | −321.3 (귀속 미측정, §1) |
| **prefill** | **2449** | **1257** | **−1192** |

## 6. 같은 변경이 decode에 준 것

| | before | after |
|---|---:|---:|
| MoE 콜 host / dsp / transport | 2714.5 / 2060.6 / 653.9 | 2019.7 / 1449.6 / 570.0 |
| gather / drain | 809.4 / 0.9+0.5 | 2.2 / 188.8+7.8 |
| 토큰당 MoE (22콜) | 59.7 ms | 44.4 ms |
| 토큰당 전체 | 84.7 ms (11.8 TPS) | 67.3 ms (14.9 TPS) |

decode의 남은 2.02 ms/콜은 CPU의 1.29보다 여전히 느리다. 남은 것은 HMX 64행 타일의
고정비 1.03 ms와 16 GB/s DMA — 문서 46 §49.3, §50.7.

## 7. 재현

```bash
# skel (커널·세션이 바뀌면 필수 — build_android.sh는 skel을 안 만든다)
HEXKL_SDK_VER=6.4.0.2 ./test/htp/build.sh
adb push test/htp/build/libnntr_hvx_skel.so <DEVICE_DIR>/
# ARM
cd Applications/CausalLM && ./build_android.sh --htp && ./install_android.sh --model=<모델>
# 실행 — TPS는 --profile 없는 빌드에서만 읽는다
NNTR_HTP_PROFILE=2 <run>
```

읽는 줄: `prefill:` / `generation:` 두 줄, `[HTP-PROFILE]`의 `M>1` 행과 `M==1` 행,
그 밑의 `weight DMA: ... first N KB took` 줄, 생성 텍스트 앞 3줄.

## 8. 아직 안 잰 것

- `2bf893f` 스크래치 상한 → prefill `alloc` 323 → ≈0 기대
- `194fffb` 버스 대역폭 투표 → `first 1024 KB took`이 16–18 GB/s에서 움직이는지
- §1의 ARM −321 ms 귀속
- 남은 최적화 목록과 기대치: 문서 46 §50.2, §50.7

## 9. 함정

- **skel을 안 다시 만들면** 이전 커널이 돈다. 이번 세션에서 첫 실행이 그랬다 — `gather`가
  안 움직였고 `first ... 0 us`로 알아챘다.
- `--profile` 빌드는 prefill을 83% 왜곡한다(문서 46 §43.4). TPS는 일반 빌드에서.
- 1회 실행이다. DSP ±4 ms 변동 위에서 0.2 ms 차이는 의미가 없다.

## 10. 다음 라운드 (코드 완료·기기 미측정) — G + A

| 커밋 | 변경 | 프로파일에서 읽을 것 | 기대 |
|---|---|---|---|
| `772a321` **G** | `hvx_quant_pack_u8_ah_mapped`의 6 MB memset을 패딩 블록만(MoE는 0 B)으로 | prefill `quant` | 1646 → ≈1200 |
| **A** | gate/up 타일을 **짝**으로 발행하고(`moe_push_gate_up_chunks`, 청크도 짝으로), 에필로그가 dequant 두 벡터에 바로 `hvx_swiglu_det_sf`를 적용해 h=silu(g)·u만 쓴다 (`hvx_dequant_swiglu_acc_tiles_to_f32`). gate/up f32는 VTCM에 존재하지 않는다. 별도 SwiGLU 패스 삭제 | prefill `swiglu` → **0**, `dequant` 1363 → ≈1600 (융합분 흡수), **합 −1.8 ms**. decode `swiglu` 24 → 0 | 콜 22.6 → ≈20.3 |

비트동일: 이전 경로는 dequant가 g·u를 VTCM에 f32로 쓰고 SwiGLU가 읽었다. 융합은 같은
intrinsic 순서로 g·u를 레지스터에 만들고 같은 `hvx_swiglu_det_sf`를 부른다 — 바이트가
같아야 하고, `swiglu_det.h`의 ARM 패리티 게이트도 그대로다. 기기 확인은 (1) 생성 텍스트가
이전 실행과 **동일**한지, (2) `unittest_hvx_mm_u8i4`의 MoE 테스트(`run_u8i4_layer_on_device.sh`).

주의: `DMA_FIRST_KB`가 이제 짝 청크(gate 16타일 + up 16타일 = 1 MB)를 세므로 `first N KB`의
N은 그대로 1024다.

### 10.1 결과 (2026-09-15, 기기) — 361 TPS, 그리고 A는 빗나갔다

```
prefill 1230 ms 360.98 TPS (직전 1257 / 353.2)    decode 15.22 TPS (14.87)    텍스트 동일
M>1  host 22158 dsp 19792 transport 2366
     [quant 1443 gather 111 requant 1161 swiglu 0 dequant 3263 acc 2959
      drain 215+49 push 47 scatter 1034 alloc 121 stage 370 mm 8925]
     first 1024 KB took 48 us = 21.6 GB/s
M==1 host 1959 dsp 1405 transport 554  [drain 165+7 mm 781 dequant 87 swiglu 0]
     first 1024 KB took 58 us = 18.2 GB/s
```

| | 예측 | 실측 | |
|---|---:|---:|---|
| A: swiglu 2055 → 0, dequant 1363 → ≈1600 | 합 −1.8 ms | **dequant 3263**, 합 −0.15 | **틀렸다.** SwiGLU는 VTCM 왕복이 아니라 HVX 계산(벡터당 ≈30 op, 계산 바닥 0.8 ms)이었고, 문서 46 §23.2에 그렇게 적혀 있었다. 메커니즘을 재지 않고 예측했다 |
| G: quant 1646 → ≈1200 | −0.4 | 1443, −0.2 | 반 |
| alloc 323 → ≈0 | | 121 | = 첫 콜 한 번의 성장 2.7 ms / 22. 나머지 콜은 0 |
| 버스 투표: first KB 30+ | | 21.6 / 18.2 (+10~20%) | **판정 불가** — 짝 청크(8 KB 행)와 같은 실행. 두 변경을 한 숫자에 겹쳤다 |

콜 22.6 → 22.2. A가 남긴 것은 시간이 아니라 **자리**다: `up` 917 KB가 비었고 SwiGLU가 풀
에필로그 안으로 들어갔다 — 둘 다 B(HVX를 HMX 뒤로)의 선행 조건이고, 빗나간 1.8 ms는
거기서 dequant 3.26 전체와 함께 숨는다.

지금 콜 = HMX 11.9 + HVX 직렬 6.9 (dequant 3.26, quant 1.44, requant 1.16, scatter 1.03)
+ transport 2.37 + 기타 1.0. 남은 레버는 B, 그 다음 transport.

DMA 속도 판별은 다음부터 **분리한다**: 아레나 프로브에 청크 모양의 2D 전송을 추가해
고립 상태에서 선형 vs 스트라이드를 재고, 투표는 별도 skel로 켜고 끈다.

## 11. B — HVX 에필로그를 HMX 뒤로 (코드 완료·기기 미측정)

콜 22.2 ms 중 HVX 직렬 6.9 ms(dequant+SwiGLU 3.26, quant 1.44, requant 1.16, scatter
1.03)가 HMX 11.9 ms와 차례로 돌았다. HMX 발행은 호출 스레드 하나가 하고, 그동안 풀
워커 4개는 논다. 이제:

| 무엇 | 어디에 숨나 | 남는 노출 |
|---|---|---|
| gate_up 에필로그(dequant+SwiGLU) 배치 ci | 배치 ci+1의 HMX 발행(47 us) — 누산기 스테이징 버퍼 2개 교대 | 블록당 마지막 배치 1개(≈18 us) |
| down 에필로그(dequant) 배치 0 | 배치 1의 HMX | 배치 1 |
| scatter(블록 b) | 블록 b+1의 활성화 DMA + gate_up 발행 | ≈0 |
| requant | **안 숨는다** — down의 HMX가 mid 전체를 필요로 한다 | 1.16 그대로 |
| quant | 안 숨는다 (루프 전) | 1.44 그대로 |

기대: dequant 3.26 → ≈0.9, scatter 1.03 → ≈0.1, **콜 22.2 → ≈19**. 레이어로 CPU 대비
1.3~1.5×.

구현:
- `hvx_worker_pool_submit/wait` — 워커만 도는 비동기 잡 하나. 호출 스레드는 참여하지
  않고(HMX를 발행해야 하니까), 워커 id k가 인덱스 k−1로 뛴다. run()과 submit()은 진행
  중인 잡을 먼저 기다린다.
- 에필로그를 잡 구조체(`hvx_dq_swiglu_job`, `hvx_dq_tiles_job`)로 — 호출자가 소유해
  submit 뒤에도 살아 있다. 커널 함수 스코프에 버퍼당 하나.
- 레이아웃: 누산기 스테이징 ×2 (+256 KB), `res_f32` 자기 영역 (+512 KB, 더는 gate를
  alias하지 않음 — 다음 블록 에필로그가 gate를 쓰는 동안 scatter가 res를 읽는다),
  `up` 영역 제거(−448 KB). 합 6.61 → **6.92 MB**, 아레나 8.3 안.
- 대기는 의존이 있는 자리에만: 배치 HMX 뒤(이전 에필로그 → 그 버퍼를 다음 배치가 씀),
  블록 마지막 에필로그(requant가 gate 전체 필요), 블록 머리(이전 scatter → res 재사용),
  콜 끝(out 복사 전), 오류 경로(`out:`).

**프로파일 읽는 법이 바뀐다:** `dequant`와 `scatter`는 이제 **노출된 대기 시간**이지
작업량이 아니다. 작업이 HMX 뒤에 잘 숨으면 0에 가깝고, `mm`은 그대로여야 한다(HMX
발행 자체는 안 바뀜). 합계가 안 맞으면 `rest`로 간다.

기기 게이트: 텍스트 동일(바이트는 안 바뀌어야 함 — 같은 연산을 같은 데이터에, 순서만
비동기), `unittest_hvx_mm_u8i4` MoE 테스트, 그리고 **여러 번** 돌려서 같은 텍스트가
나오는지 — 순서 오류는 확률적으로 나타난다. 호스트 체크는 통과하지만 스텁 submit이
즉시 실행이라 **겹침 자체는 기기에서만 검증된다.**

### 11.1 결과 (2026-09-15, 기기) — **376 TPS**, dequant는 숨었고 scatter는 안 숨었다

```
prefill 1180 ms 376.3 TPS (직전 1230 / 361)    decode 15.55 TPS (15.22)    텍스트 동일
M>1  host 19931 dsp 17548 transport 2383
     [quant 1443 gather 140 requant 1166 dequant 865 acc 2961 drain 213+49
      push 47 scatter 1056 alloc 120 stage 373 mm 8865 | rest 251]
M==1 host 1913 dsp 1348  [dequant 22.5 (87) drain 167+7 mm 770]
```

| | 예측 | 실측 | |
|---|---:|---:|---|
| dequant 3263 → ≈900 | | **865** | ✓ 에필로그가 HMX 뒤로 숨었다 |
| scatter 1034 → ≈100 | | **1056** | ✗ 대기를 **블록 머리**에 뒀다 — 앞에 있는 건 3 us짜리 활성화 대기뿐이라 숨을 데가 없었다. 첫 gate_up 배치 발행(47 us) **뒤**로 옮겼다(커밋). 기대 ≈100 |
| mm 그대로 | | 8865 | ✓ HMX 발행은 워커와 VTCM을 안 다툰다 |
| rest | 94 | 251 | 블록당 submit 7회의 깨우기 비용. 무시 |
| 콜 22.2 → ≈19 | | **19.9** | scatter 수정 후 ≈19.0 |

## 12. 남은 것 — 콜 19.9 ms의 구성과 레버

```
HMX        mm 8.86 + acc 2.96          = 11.8  (59%)   45.6 블록 × 252 us. 패딩 4.6 ms 포함, 구조 비용
transport                                2.38  (12%)
quant      1.44 (DDR 13 MB: f32 읽기 ×2 + AH 6 MB 쓰기)
requant    1.17 (down HMX가 mid 전체를 필요로 해 못 숨김)
scatter    1.06 → ≈0.1 (위 수정)
dequant    0.86 (블록당 마지막 배치의 노출)
stage/drain/gather/alloc/rest ≈ 1.1
```

| # | 레버 | 기대 | 비용 |
|---|---|---:|---|
| 1 | scatter 대기 위치 (커밋됨) | −0.9 | 0 |
| 2 | **C: 활성화를 ARM에서 u8로** — act 3.6 MB → 0.9 MB + scale/zp. DSP quant는 u8 gather만 남고 transport 바이트가 반 | quant −1.3, transport −0.9, stage −0.2 → **−2.4** | 중: ARM NEON row-quant + u8in 엔트리. `hvx_quant_rows_u8_params`와 같은 min/max 규칙이면 되고 비트동일은 불필요(DSP 양자화를 대체하는 것이지 병행이 아님). 텍스트로 확인 |
| 3 | requant의 스캔을 융합 에필로그에 — 워커별 행 min/max 부분합 → 대기 시 리듀스. 팩만 남음 | −0.5 | 중 |
| 4 | dequant 잔여 — 마지막 배치를 작게(16,16,16,8 → 이미 그렇다) | ≈0 | — |
| — | HMX 11.8 | 못 줄임 | 64행 타일 구조. HexKL이 32행 타일을 안 줌 |
| — | transport 잔여 ≈1.5 (C 후) | FastRPC 고정 + out 3.6 MB invalidate | residual을 DSP에 두는 문서 45 Phase D |

1+2+3 후 콜 ≈ **16.2 ms** = HMX 11.8 + transport 1.5 + requant 0.6 + 기타 2.3. 레이어
16.2 + 1 + 2.1–5 = 19–22 ms vs CPU 31.8 → **1.4–1.7×**. 여기가 이 커널의 실질적 바닥이다.

**전체 prefill로 보면 벽은 이제 ARM이다:** 1180 = HTP 콜 438 (37%) + ARM 742 (63%).
콜을 16.2로 내려도 1100 ms(≈404 TPS)이고, 콜이 0이어도 742 ms(≈600 TPS)다. 742 안의
MoE 몫(router/topk/staging, 레이어당 2.1–5 ms로만 알고 **안 쟀다**)은
`NNTR_M0_PROFILE=1`로 22줄 찍으면 닫힌다 — 코드 0줄. 나머지(conv 18층 · attn 6층 ·
norm)는 문서 45(HTP로)다.

## 13. prefill 레버 전체 목록 (2026-09-16) — 커널 밖까지

prefill 1180 ms = HTP MoE 콜 438 (37%) + **ARM 742 (63%)**. 커널 안은 §12로 바닥에 가깝고
(≈16 ms/콜, −80 ms), 남은 큰 덩어리는 ARM이다. ARM 742의 정체는 **안 쟀다** — 아래는
shape에서 유도한 추정이고, 첫 항목이 그걸 재는 방법이다.

### 13.1 ARM 742 ms의 추정 구성 (444 토큰)

| | GFLOP | 근거 |
|---|---:|---|
| `conv_in_proj` 18층, [444×2048]×[2048×6144] | 201 | `lfm2_causallm.cpp:128`, unit = 3·CONV_DIM |
| `conv_out_proj` 18층, 2048×2048 | 67 | |
| attn q/o 6층 2048×2048, k/v 6층 2048×512 | 56 | GQA |
| mha_core 6층 (444² × 2048) | ≈5 | 작지만 구현 속도는 미지 |
| **FC GEMM 합** | **≈ 325** | ARM 4코어 KleidiAI int4 ≈ 400–600 GFLOPS → **540–810 ms** |
| rms_norm ×44, multiply ×36, split, add, conv1d | — | 메모리 바운드, 3.6 MB × ~150회 ≈ 30–50 ms |
| MoE ARM 쪽 ×22: router dot, topk, staging memcpy 7.2 MB, `setZero` 3.6 MB | — | 2.1–5 ms/층 (§18.1의 폭) ≈ 45–110 ms |
| lm_head | — | prefill은 마지막 토큰만 (`tie_word_embedding.cpp:377`) ≈ 3 ms |

**즉 ARM 742는 거의 전부 FC GEMM 계산이고, prefill의 ARM 쪽은 대역폭이 아니라 계산
바운드다.** 같은 GEMM을 HMX가 하면: MoE 커널이 블록당 1.41 GFLOP를 252 us에 하니
5.6 TFLOPS. 325 GFLOP = **58 ms** + 에필로그·transport.

### 13.2 레버 — 기대 순

| # | 레버 | 기대 (prefill ms) | 비용 | 막는 것 |
|---|---|---:|---|---|
| **M** | **측정 먼저 (코드 0)**: `nntr_config.json`에 `"num_to_generate": 1` + `--profile` 빌드 → TYPE 합계가 곧 prefill 분해 (decode 1토큰뿐). 같이 `NNTR_M0_PROFILE=1`로 MoE ARM 쪽 22줄 | 13.1의 추정을 실측으로 | 0 | — |
| **T** | **prefill 스레드 수 (코드 0)**: `NNTR_NUM_THREADS=8`을 `num_to_generate: 1`로. §45는 decode(2 MB짜리 FC)에서 little 코어 과분할로 손해였지만, prefill GEMM은 계산 바운드라 little 코어가 보태는 쪽일 수 있다 | 600의 −20~30% = **−120~180** | 0 (효과 있으면 prefill/decode 다른 스레드 수, 작은 코드) | 측정 |
| **F1** | **`conv_in_proj` 18층을 HTP로 (prefill만)** — FC 가속 경로(`gemm_q4_0_accel_fp32`, 문서 34)는 이미 있다. `fully_connected` 레이어에 `engine=htp`를 주고, 가중치를 로드 시 bake해 **아레나의 남은 144 MiB**에 넣는다(in_proj 113 MB). decode는 `accelerates_q4_0_at_m1() == false`가 CPU Q4_0으로 돌려보내므로 그대로 | ARM −370, HTP +45 → **−320** (1180 → ≈860, ≈520 TPS) | 큼: bake→`weight_bake_export`→아레나 배치→`register_arena` 조합, 레이어 engine 배선, act 3.6 MB/out 10.9 MB 스테이징 | 아레나 여유 144 MiB — in_proj만 들어간다 |
| F2 | `conv_out_proj` + attn q/k/v/o (98 MB)를 HTP로 | −130 | F1 + 주소공간 | **DSP 주소공간이 없다** (3840 + heap 182 ≈ 4096, §40). `nntr_hvx_mem_probe_dsp_heap`으로 실제 여유를 재고, 없으면 MoE 아레나를 못 줄이니 불가 |
| K1 | 커널 §12: scatter 위치(커밋됨) + requant 스캔 융합 | −20~30 | 소~중 | — |
| K2 | C: 활성화 u8 (보류) | −53 | 중 | ARM 작업 추가 — 보류 결정 |
| K3 | transport 2.4 × 22 | −50 (C 후 −30) | 큼 (문서 45 Phase D) | |
| E | MoE ARM 쪽: staging memcpy(입력을 rpcmem 버퍼에 바로 쓰기), `setZero` 제거(커널이 zero-fill), topk 벡터화 | −30~60 | 소~중 | M0 측정 뒤 |
| N | rms_norm·multiply·split 융합 | −20~30 | 중 (nntrainer 그래프) | 낮은 우선순위 |
| A | mha_core 6층을 HTP로 (`hexkl_attn_u8` 커널이 있다) | ? (M이 정함) | 큼 | 문서 45 Phase C |

### 13.3 순서

1. **M + T** — 코드 0줄, 한 실행. 742의 구성과 스레드 효과가 나온다.
2. **F1** — 가장 큰 단일 레버. 아레나 144 MiB 안에서 in_proj만.
3. K1, E — 작은 것들.
4. F2/A — 주소공간 답이 나온 뒤.

F1까지 가면 prefill ≈ 860 ms(≈520 TPS), T가 먹으면 ≈ 700(≈630). 그 뒤는 F2/A인데
주소공간이 정한다.

## 14. HMX와 HVX를 동시에 — 콜 안의 유휴 시간 예산과 out-of-order 후보 (2026-09-16)

§11의 파이프라인은 "에필로그를 다음 배치 뒤에"까지다. 두 유닛의 유휴를 세어 보면 더 있다.
콜 19.9 ms 기준 (§11.1 실행):

```
HMX 바쁨   11.8 (mm 8.86 + acc 2.96)
HMX 유휴    4.1 = quant 1.44 (루프 전) + requant 1.17 + 마지막 에필로그 0.86 + stage 0.37 + drain 0.26
HVX 바쁨   ≈6.5 = 에필로그 3.3 + quant 1.44 + requant 1.17 + scatter 0.6
HVX 유휴   ≈8.8 — HMX가 도는 11.8 중 에필로그 3 ms만 겹친다
```

**HVX가 8.8 ms 놀고 HMX가 4.1 ms 논다.** 의존이 없는 일을 그 구멍에 넣는 후보:

| # | 후보 | HMX 임계경로에서 빠지는 것 | HVX 유휴에서 쓰는 것 | 비용 |
|---|---|---:|---:|---|
| **O1** | **꼬리 블록을 HVX GEMV로, HMX 그늘에서.** 45.6 블록 중 13.6개가 두 번째 블록(평균 10~20행)인데 HMX는 64행 값을 다 낸다(252 us). 이 블록들을 풀의 비동기 잡으로 보내고 HMX는 다음 expert로 넘어간다. 가중치는 VTCM이 아니라 **아레나(DDR)에서 직접** 읽는다 — VTCM의 A/B는 다음 expert가 곧 덮는다. DDR +75 MB/콜은 평균 9.6 GB/s인 prefill DMA에 여유가 있다. u8×i4 int32 합은 순서 무관이라 HMX 경로와 **비트동일** | **−3.4** (13.6 × 252 us) | +2.6 (블록당 ≈190 us: 5.5 MB 스트림 × (unpack + 행 수 vrmpy) / 3워커) | 큼 — 그런데 **decode의 L1(M=1 GEMV)과 같은 커널**이다. 한 번 만들어 둘 다 쓴다 |
| O2 | quant를 expert 0의 슬롯부터 팩하고 HMX를 시작, 나머지 팩은 풀에서 HMX 그늘로 | −1.3 | +1.3 | 중 (`quant_pack_worker`의 분할을 kt에서 슬롯 범위로) |
| O3 | 2블록 expert: requant(b0) 동안 HMX가 gu(b1) — h×2, mid×2 (+560 KB, 여유 1.4 MB 안) | −0.35 (13.6 블록만 해당) | 0 | 중 |
| O4 | requant 스캔(행 min/max)을 융합 에필로그의 워커별 부분합으로 → 팩만 남음 | −0.5 | 0 | 중 |
| O5 | expert 순서: 2블록 expert를 앞에 — O3의 겹침이 많아지진 않는다. 효과 없음 | 0 | | 안 함 |

O1–O4 합 ≈ **−5.5 ms/콜 → 콜 19.9 → ≈14.4, prefill −120 ms.** HMX 바닥 11.8에 거의 닿는다
(그 뒤는 transport와 stage뿐). O1이 절반이고 decode 커널을 겸하므로 먼저다.

**커널 밖의 out-of-order — 청크 prefill.** 444 토큰을 반으로 갈라 DSP가 앞 반의 MoE(L)를
도는 동안 ARM이 뒤 반의 conv(L+1)… 를 돌리면 ARM 742와 DSP 438이 겹친다. 그런데 MoE HMX는
패딩 지배라 M을 반으로 줄이면 블록 수가 45.6 → 2×36 = 72로 **+58%** 늘어난다(expert마다
패딩). 겹침으로 얻는 것보다 잃는 게 클 수 있어 **M 측정 뒤에 산술로 판단**한다. 적어 둔다.

### 14.1 프로파일링 절차 (이번 라운드)

```bash
# ARM: --profile 빌드 (노드별 타이머). skel은 7527b15 그대로
cd Applications/CausalLM && ./build_android.sh --htp --profile && ./install_android.sh --model=<모델>

# 1) prefill 분해 — <model_dir>/nntr_config.json 의 "num_to_generate" 를 1 로
NNTR_M0_PROFILE=1 NNTR_HTP_PROFILE=2 <run> 2>&1 | tee prefill_prof.log
NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=2 <run> 2>&1 | tee prefill_t8.log      # 스레드 효과 (T)

# 2) decode 분해 — num_to_generate 원래대로(512)
NNTR_HTP_PROFILE=2 <run> 2>&1 | tee decode_prof.log
NNTR_HTP_PROFILE=3 <run> 2>&1 | tee decode_p3.log                         # transport 정체 (측정 B)
```

읽는 것:

| 블록 | 열 | 뜻 |
|---|---|---|
| `[PROFILE]` 표 (`key avg min max sum pct`) | `sum`, `pct` | 노드/TYPE별 합. `num_to_generate: 1`이면 decode 1토큰(≈65 ms)만 섞이므로 **sum ≈ prefill**. `fully_connected`·`mha_core`·`rms_norm`·`lfm2_moe`·`causal_conv1d`·`custom_multiply`를 본다 |
| `[M0-PROF] moe_layer[i] tokens=444 us= setup= router= topk= wksp= gather= ffn= route= scatter= other=` × 22 | `ffn` vs 나머지 | `ffn` ≈ HTP host + staging; **나머지 합**이 §13.1의 "MoE ARM 쪽 2.1–5 ms" |
| `[HTP-PROFILE] M>1` | `host`, `scatter` | 7527b15 반영 확인: scatter ≈100 |
| `prefill:` 줄 | TPS | **`--profile` 빌드는 prefill을 83% 왜곡** — 두 실행 사이의 **비율**만 읽는다 (T의 효과) |
| decode `[PROFILE]` | `output_of_causallm` avg, `fully_connected` sum/512, `mha_core` | 문서 48 §2 ②③의 실측 |
| decode `PROFILE=3` `M==1` | `transport` | 565 → ≤300이면 wake/클록, 아니면 마샬링 |

돌려보낼 것: 네 로그의 `[PROFILE]` 표 전체, `[M0-PROF]` 22줄, `[HTP-PROFILE]` 블록, `prefill:`/`generation:` 줄.

## 15. `--profile` 노드 표 읽기 (2026-09-16, decode 512 실행) — 놀란 것 셋

`num_to_generate: 512`, `--profile` 빌드. 표는 prefill 1회 + decode 512회 합이라 **노드의
`max` = prefill 콜**, `sum − max` ≈ decode 512회. (`[M0-PROF]`는 이 실행엔 없다 —
prefill-only 실행이 따로 필요하다.)

### 15.1 prefill (nn_forward 1441 ms, 프로파일 빌드) — max 열 합계

| | ms | 비고 |
|---|---:|---|
| `lfm2_moe` ×22 | **714** | HTP host ≈ 438 → **ARM 쪽 ≈ 276 = 12.5 ms/층** |
| `fully_connected` | 482 | conv_in_proj 212, layer0/1 dense FFN 98, conv_out_proj 76, attention_out 46, wq 28, wk/wv 22 |
| `mha_core` ×6 | 116 | layer2가 55 ms(첫 콜 효과?) — 빼면 61 |
| `output_of_causallm` | 64 | **lm_head twin의 첫 콜 빌드**(75 MB repack). 1회성인데 prefill 안에서 |
| norm·add·mul·split·conv1d·swiglu | ≈70 | |

**놀란 것 ①: MoE 레이어의 ARM 쪽이 12.5 ms/층이다.** §13.1은 2.1–5로 잡았다. 22층이면
276 ms — 프로파일 빌드 기준이지만 스케일해도 ≈225, FC 다음으로 큰 덩어리다. 무엇인지는
`[M0-PROF]`(setup/router/topk/wksp/gather/ffn/route/scatter/other)가 말한다. 후보:
`output.setZero()` 3.6 MB, 워크스페이스 Tensor 4개 생성, staging memcpy 7.2 MB, topk.
**놀란 것 ②:** attention 층 뒤의 MoE(2·6·10·14·18·21)는 23–27 ms, conv 층 뒤의 MoE는
30–42 ms. 같은 코드다. 라우팅 차이로는 10 ms가 안 나온다 — M0가 갈라야 한다.
**놀란 것 ③:** lm_head twin이 첫 prefill에서 64 ms를 빌드한다. 로드 시점으로 옮겼다
(`prepareLmhead`, `repack_weight`에서 호출 — 커밋). **prefill −52 ms(비프로파일 환산).**

### 15.2 decode (토큰당, 512회 평균)

| | ms/token | |
|---|---:|---|
| `lfm2_moe` 22층 | 45.1 | 콜 avg 2.05; HTP host 1.91 → ARM 쪽 0.14/층 |
| `fully_connected` | 8.9 | 170 MB, 19 GB/s |
| `output_of_causallm` | **4.3** | twin 확인: 25.7 → 4.3 (min 2.5). 75 MB → 17.6 GB/s |
| `mha_core` 6층 | 3.2 | 530 us/층 — KV 2 MB 읽기에 비해 느리다 |
| 나머지 노드 | 2.0 | |
| **표 밖** (샘플링·토크나이저·argmax 65536) | **3.7** | 35146 − 33271 ms |
| 합 | ≈ 67 | 15 TPS |

문서 48 §2의 ②③이 실측됐다: lm_head 4.3(예상 3.4), ARM 나머지 = FC 8.9 + mha 3.2 +
기타 2.0 + 표 밖 3.7 = **17.8**.

### 15.3 이 표가 바꾼 순위

prefill: **M0 측정(코드 0) → E(MoE ARM 쪽, ≈225 ms — 12.5 ms/층의 정체에 따라 소~중)
→ F1(conv_in_proj, 212 ms)**. E가 F1과 같은 급으로 올라왔고 훨씬 싸다.
decode: 표 밖 3.7 ms/token(샘플링/argmax)은 커널 밖의 공짜에 가까운 6%다.

### 15.4 `NNTR_NUM_THREADS=8` vs 기본(=hardware_concurrency/2 = 4) — 같은 실행, 두 번째 표

| | 4 스레드 | 8 스레드 | |
|---|---:|---:|---|
| prefill `nn_forward` | 1441 | **1347** | −94 ms (−6.5%) |
| ├ `lfm2_moe` ×22 (max 합) | 714 | 700 | ARM 쪽이 스레드에 안 움직인다 → 단일 스레드 작업(memset/memcpy/topk) 가능성 ↑ |
| ├ `fully_connected` | 482 | 472 | conv_in_proj 212→**152**, 그런데 conv_out_proj 76→110 |
| ├ `mha_core` ×6 | 116 | 88 | |
| ├ `output_of_causallm` 첫 콜 | 64 | 69 | 둘 다 twin-at-load 커밋 이전 |
| decode ms/token ((총−prefill)/512) | 65.8 | **69.3** | +3.5 |
| ├ `lfm2_moe` | 45.1 | 44.8 | HTP라 무관 |
| ├ `fully_connected` | 8.9 | **16.4** | 아래 |
| ├ `output_of_causallm` | 4.3 | 2.7 | 75 MB → 27 GB/s |
| ├ `mha_core` | 3.2 | 2.4 | |

**decode FC가 두 배가 된 이유는 표 안에 있다.** 같은 모양의 `conv_in_proj`(6 MB, 4스레드
≈ 250 us)가 층마다 다르다: layer23 430 us, layer12 593, layer3 **1010**, layer0 **1558**
(min은 전부 124–136). 호출당 일이 250 us인데 8개 스레드를 깨우고 — little 코어가
끼면 — 제일 느린 코어가 꼬리를 잡는다. layer0(토큰의 첫 CPU 연산, 스레드가 다 자고 있다)
과 layer3(attention 층 직후)이 가장 나쁘다. prefill은 행렬이 커서 8개가 이기지만
conv_out_proj(2048×2048)는 거기서도 졌다.

**결론:** 스레드 수는 prefill/decode 트레이드오프다. 전역 하나로 두면 **4 유지**(decode
+3.5 ms/token > prefill −94 ms는 512 토큰 생성에서 손해). T 레버는 "prefill 8 / decode 4"
로 단계별로 바꿀 수 있을 때만 −90 ms이고, 그 전에 big 코어 4개에 고정하는 게 같은 값을
공짜로 줄 수도 있다(미측정).

## 16. prefill 레버 목록 v2 (2026-09-16, §15의 실측 반영) — 이것만 보면 된다

기준: 비프로파일 1180 ms / 444 토큰 / 376 TPS. 구성(프로파일 표에서, ms):

```
HTP MoE 콜 22        438   (19.9/콜: HMX 11.8 구조비 + transport 2.4 + 에필로그 노출 ≈5.7)
MoE ARM 쪽 22        262–276  ← 정체 미측정. 스레드 수에 안 움직였다(714→700)
fully_connected     472–482  = conv_in_proj 152–212 + dense FFN(l0,l1) 66–98 + conv_out_proj 76–110 + attn q/k/v/o ≈140
mha_core 6           88–116  (layer2가 34–55: 첫 콜?)
lm_head twin 빌드     64–69  → 로드 시점으로 옮김 (b616248, 미측정)
norm/mul/split/add   ≈70
```

코드에서 확인한 것 두 가지(`lfm2_moe_layer.cpp:762-824`): HTP 경로에서도 ①
`output.setZero()` 3.6 MB를 하고(커널이 다시 zero-fill한다) ② prefill 워크스페이스
Tensor 4개를 만든다 — `FloatTensor`의 `new float[n]{}`는 **zero-fill**이라 4–17 MB
memset + page fault를 층마다 내고 안 쓴 채 버린다. 둘 다 `tryMoeLayerOnAccelerator`
뒤로 옮기면 된다. 이것이 12 ms/층 전부인지는 M0가 말한다(staging memcpy는 아니다:
`[HTP-PROFILE] arm staging memcpy` 총 20 ms/실행, prefill 몫 ≈10).

| 순위 | 레버 | 기대 (ms) | 코드 | 상태·근거 |
|---|---|---:|---|---|
| 0 | **M0 측정**: `num_to_generate: 1` + `NNTR_M0_PROFILE=1 NNTR_HTP_PROFILE=2` | E의 크기를 정함 | 0 | 표 없이 E를 고르면 추측 |
| 1 | **E0 lm_head twin을 로드로** | **−52** (프로파일 −64) | 커밋 b616248 | 앱 재빌드 후 `output_of_causallm` max가 2–5 ms면 확인 |
| 2 | **E1 MoE ARM: setZero·워크스페이스를 가속 실패 뒤로** | −10 ~ −40 (memset·fault 몫) | 두 블록 이동 | 코드로 확정된 낭비. M0의 `setup`·`wksp`가 값을 준다 |
| 3 | **E2 MoE ARM 나머지** (router dot·topk·`other`) | 12 ms/층 중 E1 뺀 만큼, 최대 **−200** | M0 결과에 따라 소~중 | attn 층 뒤(24 ms)와 conv 층 뒤(31–37)가 다른 이유도 여기서 |
| 4 | **F1 conv_in_proj 18층을 HTP로** | −100 ~ −155 (ARM 152–212 → HTP ≈55: 36 계산 + 15 staging) | 큼: bake→아레나 144 MiB(in_proj 113 MB)→`engine=htp` 배선 | FC 가속 경로(문서 34)는 있다. decode는 CPU 유지 |
| 5 | **O1 꼬리 블록을 HVX GEMV로** (§14) | −75 (3.4/콜) | 큼 — decode GEMV와 같은 커널 | HMX 13.6 블록의 252 us 고정비 제거. 비트동일 |
| 6 | O2+O3+O4 (quant 분할·2블록 겹침·requant 스캔 융합) | −45 (2.1/콜) | 중 | §14 표 |
| 7 | **T 스레드 8 (prefill만)** | −94 | 단계별 스레드 수 (작음) 또는 big 코어 고정 | §15.4: 전역 8은 decode +3.5 ms/token이라 불가 |
| 8 | mha layer2 첫 콜 34–55 ms의 정체 | −20 ~ −40 | 아마 첫 콜 할당 → 로드로 | 나머지 5층은 8–15 |
| 9 | A mha_core 6층을 HTP로 (`hexkl_attn_u8`) | −40 ~ −60 | 큼 (문서 45 Phase C) | 8 뒤에 |
| 10 | F3 dense FFN l0/l1을 HTP로 (`gemm_qs4cx_fused_swiglu` 경로 있음) | −50 ~ −70 | 중 — 가중치 44 MB, F1과 아레나 경합(113+44 > 144) | F1과 택일 또는 아레나 재배치 |
| 11 | N norm·multiply·split 융합 | −20 ~ −30 | 중 (그래프) | 낮음 |
| 12 | K3 transport 2.4/콜 (residual을 DSP에) | −30 ~ −50 | 큼 (문서 45 Phase D) | |
| — | F2 conv_out_proj·attn proj를 HTP로 | −130 | — | DSP 주소공간 없음 (§13.2) |
| — | C 활성화 u8 (ARM) | −53 | — | **보류** (사용자) |
| — | 청크 prefill로 ARM/DSP 겹치기 | ? | — | 패딩 +58%로 손해 가능성. 산술 뒤 |

**순서:** 0 → 1·2(같은 빌드에서 측정) → 3 → 4 → 5·6 → 7·8. 1–3이 다 먹으면 1180 →
≈900(≈490 TPS), 4까지 ≈770(≈575), 5·6까지 ≈650(≈680). 숫자는 기대이고 매 단계 측정으로
갱신한다.

## 17. E0 + E1 결과 (2026-09-16, 기기, `--profile` 빌드, decode 512, 기본 4스레드)

| | 이전 (§15.1) | E0+E1 | |
|---|---:|---:|---|
| prefill `nn_forward` | 1441 | **1294** | −147 |
| `lfm2_moe` ×22 (max 합) | 714 | **458** | **−256**. HTP host 438이면 ARM 쪽 ≈ 20 = **0.9 ms/층** |
| ├ attn 층 뒤 / conv 층 뒤 | 24 / 31–37 | 18–22 / 18–22 | 비대칭 소멸 — 페이지 폴트였다 |
| ├ layer2 (첫 콜) | 39 | 30.6 | 첫 콜 +10은 남음 (스크래치 1회 성장 등) |
| `output_of_causallm` 첫 콜 | 64 | **5.6** | E0 확인. avg 4.3, min 2.5 |
| `fully_connected` | 472–482 | **580** | conv_in_proj 212 → 260. 아래 |
| `mha_core` ×6 | 88–116 | 166 | layer2 58 |
| decode ms/token | 65.8 | 65.5 | 손대지 않았다 |

**E1이 12 ms/층의 거의 전부였다.** 4개 워크스페이스 Tensor의 zero-fill 할당(`new float[n]{}`)
+ 새 매핑의 페이지 폴트 + `setZero` 3.6 MB — 매 층 만들고 안 쓰고 버리던 것. §16이 −10~−40으로
잡았는데 −256이 나왔다: memset 바이트가 아니라 **mmap/munmap + 페이지 폴트**가 값이었고,
attention 층 뒤가 덜 느렸던 것도 그 직전 해제 패턴의 차이였을 것이다. 산술로 못 잡는 종류다.

**이 실행은 CPU 쪽 잡음이 크다.** FC가 100 ms, mha가 50–80 ms 나빠졌고, 사소한 노드에 이상치가
있다: `layer21_attention_norm` max 15 ms, `layer4_conv_mul_pre` 17.7, `layer23_conv_mul_post`
10.7, `input` 19.5, `cache_k_l2` 8. 노름 하나가 15 ms면 코드가 아니라 **선점/코어 이동/열
쓰로틀**이다. 잡음을 빼면 이 빌드의 prefill은 ≈ 1100 (프로파일 빌드)로 본다. 비프로파일
`prefill:` TPS로 확정해야 한다.

**구성 (이 실행, ms):** FC 580 (45%) · MoE HTP 458 (35%) · mha 166 · 원소 ≈85 · lm_head 6.
MoE ARM 쪽은 끝났다. 다음은 FC이고, 순서는 **T(단계별 스레드 수, 코드 소) → F1(conv_in_proj
HTP, 코드 대)**. T: `ThreadManager::parallelize`가 `compute_workers_.size()+1`로 나누므로
런타임 상한(`active` 캡) 하나면 prefill 8 / decode 4가 된다 — 8스레드 prefill FC 472 vs
4스레드 580(잡음 포함)/482.

## 18. 범위 고정: FFN만 (2026-09-16) — 다른 레이어는 CPU 고정, 비교 실험 조건

사용자 결정: FFN 외 레이어는 **CPU에 둔 채** FFN(HTP) vs FFN(CPU)를 비교한다. 그러므로 F1·F2·F3·A(다른
레이어를 HTP로)와 T(스레드)는 제외. 남는 것은 MoE 경로 458 ms(HTP 콜 438 + ARM 20) 안의
레버뿐이다. 콜 19.9 ms = HMX 11.8 (mm 8.86 + acc_read 2.96) + transport 2.4 + 에필로그 노출 ≈5.7.

| 순위 | 레버 | 콜당 | prefill | 코드 | 비고 |
|---|---|---:|---:|---|---|
| 1 | **O1 꼬리 블록을 HVX GEMV로, HMX 그늘에서** (§14) | −3.4 | **−75** | 큼 | 13.6 블록 × 252 us 고정비. decode GEMV(L1)와 같은 커널 |
| 2 | 첫 콜 +10 ms (layer2 30.6 vs 18–22) 정체 | 1회 | −10 | 측정 후 소 | `NNTR_HTP_PROFILE=2`의 첫 콜 stage 열이 말한다 |
| 3 | O2 quant를 슬롯 단위로 나눠 HMX 조기 시작 | −1.3 | −29 | 중 | |
| 4 | O4 requant 스캔을 융합 에필로그의 워커 부분합으로 | −0.5 | −11 | 중 | |
| 5 | O3 2블록 expert의 requant 뒤에 HMX 겹침 | −0.35 | −8 | 중 | |
| 6 | K3 transport 2.4 → ≈1.5: act/out을 rpcmem 텐서로 두어 staging 제거, 또는 residual을 DSP에 | −0.9 | −20 ~ −50 | 중~큼 | 문서 45 Phase D |
| 7 | ARM 잔여 0.9 ms/층 (router dot·topk의 토큰별 vector 할당) | — | −10 | 소 | M0의 `topk`/`other`로 확인 뒤 |
| ? | **acc_read 2.96 ms/콜을 다음 타일의 mm 뒤에 숨기기** | 최대 −2.96 | 최대 −65 | 확인 필요 | 우리가 쓰는 HexKL micro API는 clear/mm/acc_read뿐. 빌드 머신의 실제 `hexkl_micro.h`에 accumulator 뱅크 선택(두 번째 acc) 변형이 있는지 봐야 한다. 없으면 0 |
| — | C 활성화 u8 (ARM) | −2.4 | −53 | 중 | **보류** |
| — | HMX mm 8.86 | 못 줄임 | | | 64행 타일 구조, 5.6 TFLOPS 실효 |

1–7 합 ≈ **−165 ~ −195 ms**: MoE 경로 458 → ≈270, 콜 19.9 → ≈14. 비프로파일 prefill ≈ 1180 − 147(E0·E1)
− 180 ≈ **850 ms(≈520 TPS)**. FFN 층 자체로는 CPU 31.8 ms/층 대비 HTP 20.8 → 12.3, **1.5× → 2.6×**.
순서: **1 → 2 → 3·4·5 → 6 → 7**. O1이 반이고 decode 커널을 겸하므로 먼저.

## 19. O2 — 활성화 팩을 HMX 뒤로 (코드 완료 · 기기 미측정)

**바꾼 것 셋.**
1. `hvx_worker_pool`에 **백그라운드 레인**: `submit_bg(func, ctx, n_units, done)` /
   `wait_bg(n)`. 워커는 run/submit 잡이 없을 때 유닛을 하나씩 CAS로 가져가고, 유닛 하나 뒤에
   다시 포그라운드를 본다(포그라운드 submit이 기다리는 최대치 = 유닛 하나 ≈ 10 us). 기다리는
   호출자도 유닛을 가져간다. 잡 필드는 **claim 뒤에** 읽어 이전 잡의 stale `bg_n`으로 들어온
   워커도 제 잡을 본다(구조체 주석). 호스트 검사 `worker_pool_host_check.c`(pthread stub
   `stub/qurt.h`)가 두 레인 동시·재사용(큰→작은→큰)·인라인 경로를 돈다.
2. `hvx_quant_pack_u8_ah_block`: 64행 블록 하나의 mapped 팩. 기존 k-tile 분할 팩과 같은
   `quant_pack_group4` 본체를 쓰므로 바이트 동일.
3. 커널: 팩을 슬롯 블록 단위 bg 유닛으로 제출하고, **블록을 DMA 큐에 넣기 직전에만** 그
   블록을 기다린다. expert 0의 gate_up 푸시를 scan 앞으로 옮겨 DMA_FIRST가 scan 뒤에 숨는다.

**기대:** QUANT 1.44 → ≈0.45(scan) + 대기 잔여, DMA_FIRST ≈0. 콜당 −1.0~−1.3, prefill −22~−29.
**볼 것:** `[HTP-PROFILE] M>1` 표의 quant·dma_first·dequant(bg 유닛이 에필로그를 지연시키면
여기가 오른다)·host. 출력은 바이트 동일해야 한다(팩 산술 불변) — 텍스트 비교.

### 19.1 결과 (2026-09-16, 기기, 비프로파일, `NNTR_NUM_THREADS=8`) — **480 TPS**

```
prefill: 444 tokens, 924 ms, 480.5 TPS      (§11.1 376 TPS, 1180 ms → −256 ms)
generation: 512 tokens, 25431 ms, 20.1 TPS  (15.55 → 20.1)
```

E0 + E1 + O2가 한 빌드에 들어간 첫 비프로파일 수치. 8스레드였으므로 T의 몫(프로파일 빌드에서
−94)이 섞여 있고, 셋의 개별 몫은 이 로그로는 못 가른다. decode 20.1 TPS는 프로파일 빌드가
말한 "8스레드는 decode 손해"와 반대다 — 노드 타이머가 스레드 깨우기 비용을 부풀렸을 수 있다.
비프로파일 4스레드 한 번이 갈라 준다. O2의 바이트 동일성은 텍스트 비교 대기.

**500 TPS까지 −40 ms.** 남은 FFN 범위 레버(§18)에서: O1 −30~−55(bg 레인이 팩 1 ms를 이미
쓰므로 유휴 8.8 ms 중 ≈7.5가 남는다), transport 2.4/콜의 정체(PROFILE=3으로 wake/클록인지
마샬링인지), O4 −8, O3 −8, 첫 콜 −10.

### 19.2 결과 (2026-09-16, 기기, `NNTR_HTP_PROFILE=2`/`3`, decode 512 실행의 M>1 행)

| us/콜 | §11.1 (B 이후) | O2 (PROFILE=2) | PROFILE=3 (5회 중 최소) | |
|---|---:|---:|---:|---|
| host | 19900 | **18258** | 17663 | **−1.65 ms/콜 → prefill −36** |
| dsp | 17500 | 15898 | 15777 | |
| transport | 2400 | 2360 | **1886** | 첫 회분 0.47 = act 3.6 MB의 dirty writeback으로 추정. 구조적 1.9 남음 |
| quant | 1440 | **354** | 341 | O2 확인: 팩 1.0 ms가 HMX 뒤로 갔다 |
| dma_first | (측정) | **1 us** | 1 us | scan 뒤에 숨었다 |
| scatter | 1056 | **23** | 25 | 7527b15 확인 |
| dequant | 860 | **1178** | 1222 | **+0.32**: 64행 bg 유닛이 에필로그 픽업을 늦춘다 → 16행 유닛으로 (커밋) |
| requant | 1170 | 1248 | 1265 | |
| acc_read | 2960 | 2985 | 2972 | |
| mm | 8860 | 8900 | 8918 | |
| gather / stage / drain / alloc / rest | | 228 / 369 / 161+47 / 109 / 249 | 238 / 385 / 76+36 / 0.5 / 248 | alloc·drain의 첫 콜 몫 ≈ 0.2 ms |

비프로파일: **8스레드 924 ms (480 TPS), 4스레드 1039 (427)** — decode도 20.1 vs 18.5 TPS로 8이
낫다. 프로파일 빌드의 "8스레드 decode 손해"는 노드 타이머의 왜곡이었다. 코드는 안 바꾸고
`NNTR_NUM_THREADS=8`로 돌린다. 세 실행의 출력 텍스트는 동일.

**500까지 −36 ms (8스레드 기준).** dequant 회복 −0.3/콜(16행 유닛, 미측정) + O1 −2~−2.5/콜.
transport의 구조적 1.9 ms는 act/out 3.6 MB의 캐시 유지비로 보이며 바이트를 줄여야 움직인다(C 보류).

## 20. 레버 목록 v3 (2026-09-16, O2 프로파일 기준) — prefill과 decode, 범위 표시

기준 콜(prefill, us): mm 8900 · acc_read 2985 · transport 2360 · requant 1248 · dequant 1178 ·
stage 369 · quant 354 · rest 249 · gather 228 · drain 208 · alloc 109 · push 47 · scatter 23 = host 18258.
prefill 924 ms(8스레드) = HTP 콜 ≈400 + ARM ≈520.

### 20.1 prefill, FFN 범위 안

| # | 레버 | 콜당 | prefill | 코드 | 상태 |
|---|---|---:|---:|---|---|
| 1 | dequant 회복: 팩 유닛 16행 | −0.3 | −7 | 커밋 `1187884` | 미측정 |
| 2 | **O1 꼬리 블록을 HVX로** (bg 레인 위) | −2~−2.5 | **−45~−55** | 큼 | 진행 중 |
| 3 | **acc_read를 다음 타일 mm 뒤에** — HexKL에 2번째 accumulator가 있을 때만 | ≤ −2.98 | **≤ −65** | 중 | `hexkl_micro.h` 목록 대기 |
| 4 | O4 requant 스캔을 융합 에필로그의 워커 부분합으로 | −0.5 | −11 | 중 | |
| 5 | O3 2블록 expert: requant(b0) 뒤에 HMX gate_up(b1) — gate/mid ×2 | −0.35 | −8 | 중 | |
| 6 | 첫 콜 워밍업: 로드 끝에 더미 콜 1회 (layer2의 +10 ms, alloc·drain 첫 몫) | 1회 | −10 | 소 | |
| 7 | gather 228: 다음 블록의 act DMA를 현재 블록 down 중에 미리 | −0.2 | −4 | 소 | |
| 8 | transport 첫 회분 0.47 (act 3.6 MB dirty writeback 추정) — staging을 non-temporal 저장으로 | −0.3? | −7? | 소 | 가설, 측정 필요 |
| 9 | transport 구조적 1.9: 바이트를 줄여야 움직인다 — act u8(C, 보류) −0.9, out은 f32 유지 | −0.9 | −20 | 중 | **보류** |
| 10 | **청크 prefill로 ARM/DSP 겹치기** — 444 토큰을 2청크, 청크1의 MoE(DSP) 동안 ARM이 청크2의 conv/attn. 계층 이동 없음, 겹침만 | — | **−200~−300** | 큼 (그래프 실행을 청크 파이프라인으로, FastRPC 비동기) | 패딩 +58%(45.6 → 72 블록, +3 ms/콜)가 대가. O1 뒤엔 그 대가가 작아진다. 산술: max(ARM 520, DSP 400+70) ≈ 650 ms → **≈680 TPS** |
| — | mm 8900 | 못 줄임 | | | 64행 타일 구조 |

### 20.2 decode, FFN 범위 안 (20.1 TPS = 50 ms/token, MoE ≈ 40 = 1.8 ms/콜: dsp 1.25–1.38 + transport 0.36–0.45)

| # | 레버 | 토큰당 | 코드 | 비고 |
|---|---|---:|---|---|
| D1 | **DMA 처리량**: 21.5 MB/콜을 16 GB/s로 읽는다(1.34 ms ≈ dsp 전부). 고립 측정은 38.8. 다중 dmstart·descriptor 크기·bus vote를 아레나 프로브로 (문서 48 측정 C) | **−10~−15 ms** | 소~중 (프로브 먼저) | decode의 벽 1 |
| D2 | **transport 0.4 ms × 22 = 9 ms/token (18%)**: FastRPC 콜당 비용. 레이어 22개를 한 콜로(다른 레이어가 CPU라 불가) 대신 **persistent DSP 워커 + 공유 메모리 큐(dspqueue)** | −6~−7 | 큼 | decode의 벽 2 |
| D3 | M=1 GEMV(L1): HMX 64행 패딩(mm 767 + acc 260 = 1.03 ms) → HVX GEMV. O1과 같은 커널. DMA 벽(D1) 아래로는 못 내려간다 | −0~−5 (D1 뒤에 의미) | O1과 공유 | |
| D4 | gather 108–132 us: 4 expert의 act 블록 DMA 대기 | −2 | 소 | D3에 흡수 |

### 20.3 범위 밖 (기록만)

F1 conv_in_proj HTP(−100~−155) · F3 dense FFN HTP(−50~−70) · A mha HTP(−40~−60) · T 단계별 스레드(env로 대체) · C ARM u8 활성화(−53, 보류) · N 원소 융합(−20~−30) · decode의 ARM 몫(FC 8.9, lm_head 2.7, mha 2.4).

**순서 제안:** prefill 2 → 3(헤더 확인 즉시) → 4·5·6·7 → 10. decode D1 프로브 → D2.
