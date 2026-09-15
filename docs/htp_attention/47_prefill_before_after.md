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
