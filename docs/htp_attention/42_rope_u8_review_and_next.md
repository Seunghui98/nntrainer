<!-- SPDX-License-Identifier: Apache-2.0 -->

# 42 -- `f21d0ad`/`339041c`/`d7b61e9` 리뷰, 그리고 다음에 무엇을 할 것인가

`40_rope_u8_task.md`가 지시한 커널이 구현됐고(`f21d0ad`), QNN u8 테이블
캐시가 붙었고(`339041c`), per-op 확장 핸드오프 문서가 나왔습니다(`d7b61e9`,
`41_rope_u8_qnn_per_op_handoff.md`).

이 문서는 그 세 커밋을 코드를 읽고 리뷰한 결과와, **doc 41의 작업 순서를
뒤집자는 제안**입니다. 디바이스에서 돌려보지 못했습니다 -- 아래 §1의 결함은
전부 코드를 읽어서 찾은 것이고, §1.1은 재현 방법을 같이 적었습니다.

---

## 0. 먼저, 이전 문서의 사실 정정

`ref_16` 부록이 열거한 트레이스 필드에 `Step Size` / `Zero Offset`이 없어서
"optrace에는 scale/zero-point가 없다"고 쓴 적이 있습니다. **틀렸습니다.**
doc 41 §"Facts observed in the QNN trace"가 실제 값을 인용하고 있습니다:

```
Data Type  : QUInt8
Step Size  : 0.00784314
Zero Offset: 128
```

부록은 그 분석이 *쓴* 필드만 열거한 것이지 전체 스키마가 아니었습니다.
per-op 메타데이터 추출은 가능하고, doc 41이 "tensor ID를 따라가야 한다"고
쓴 것이 맞는 진단입니다.

다만 §1.3과 §2.1이 이 값에 대한 별개의 문제를 지적합니다.

---

## 1. 반드시 고쳐야 하는 것 (device 재실행 전)

### 1.1 `hvx_rope_u8.c:96-98` -- 버퍼 밖 읽기/쓰기

```c
for (uint32_t w = 0; w + dim <= width; w += dim) {
  for (offset...) rope_block(row + w, out + w, ...);
  if (dim < width) {
    memcpy(out + w + dim, row + w + dim, width - dim);   /* <-- */
  }
}
```

`out`은 **row의 base**이지 `out + w`가 아닙니다. 마지막 반복
`w = width - dim`에서 목적지는 `out + width` -- 정확히 row의 끝 -- 이고
길이는 `width - dim`입니다.

Q의 실제 shape(`width = 2048`, `dim = 128`, Qwen3-0.6B nHq=16)에서:

```
마지막 반복 w=1920:  memcpy(y + m*2048 + 2048, x + m*2048 + 2048, 1920)
m = n_rows-1 이면    y/x 버퍼 끝에서 1,920 바이트 밖
```

FastRPC ION 버퍼입니다.

**출력은 맞습니다.** 각 memcpy가 쓰는 영역을 다음 반복이 다시 회전해서
덮기 때문입니다. 그래서 **출력 비교 테스트로는 절대 안 잡힙니다** -- 조용히
메모리만 넘어갑니다.

**재현:** `width/dim >= 2`이고 `width % dim == 0`인 케이스 하나면 됩니다.
현재 테스트 3개(`{1,64,64}`, `{3,96,64}`, `{2,128,128}`)는 전부
`dim == width`이거나 `width % dim != 0`이라 이 조건에 안 걸립니다 (§3).
`{2, 512, 64}`를 추가하고 ASAN 호스트 하네스나 device에서 y 뒤에 가드
바이트를 두면 즉시 보입니다.

**고칠 방법: 이 `if` 블록을 지우세요.** 두 가지 이유로 필요가 없습니다.

1. `40_rope_u8_task.md` §1.4: partial rotary는 `_compute_proportional_parameters`
   (`mha_core.cpp:959`)가 `thetas[i] = 0`으로 처리하므로 그 lane은 테이블에서
   항등 회전을 받습니다. 커널에 "회전 안 하는 채널" 경로가 있으면 안 됩니다.
2. `width = n_heads * head_dim`, `dim = head_dim`이므로 `width % dim == 0`이
   항상 참입니다. 남는 꼬리가 없습니다.
3. skel이 이미 `memcpy(y, x, expected_x)`를 먼저 합니다
   (`nntr_hvx_rope.c:156`). 설령 꼬리가 있어도 이미 복사돼 있습니다.

llama.cpp `rope-ops.c`의 `if (rctx->n_dims < ne0) hvx_copy_f32_uu(...)`를
옮긴 것으로 보이는데, 저쪽은 `ne0`가 **row 전체 폭**이고 `n_dims`가 회전
접두부라서 **row당 한 번** 도는 코드입니다. 여기 루프는 row 안의 head
세그먼트를 도는 것이라 의미가 다릅니다.

### 1.2 `nntr_hvx_rope.c:148-149, 186-187` -- DSP 스택을 넘는 VLA

```c
float row_scale[n_rows];
int32 row_zp[n_rows];
```

`n_rows`는 4096까지 허용됩니다 (`:125`). **16 KB + 16 KB = 32 KB**입니다.
이 트리의 QuRT 워커 스레드 스택은 16 KB이고
(`hvx_worker_pool.c:38`, `HVX_WORKER_POOL_STACK_SIZE`), FastRPC 서비스
스레드의 기본값도 같은 자릿수입니다. 넘치면 예외가 아니라 **조용한 손상**
입니다. 두 엔트리 포인트 모두 해당됩니다.

**고칠 방법: 배열을 만들지 마세요.** broadcast만 처리하면 되므로 커널에
stride를 넘기는 게 한 줄입니다:

```c
/* s_in_stride/zp_in_stride: 0 = broadcast, 1 = per-row */
... s_in[(size_t)m * s_in_stride] ...
```

`hvx_rope_u8_rows`에 파라미터 2개를 추가하고 skel은 `s_inLen == 1 ? 0 : 1`을
넘깁니다. 할당도, 복사도, 스택도 없습니다.

### 1.3 테이블 스케일이 트레이스 값과 다릅니다

`nntr_hvx_rope.c:21` -- `#define ROPE_QNN_TABLE_SCALE (1.0f / 127.0f)`

```
1/127            = 0.007874016
1/127.5 = 2/255  = 0.007843137
트레이스 관측값   = 0.00784314
```

**관측값은 2/255입니다.** doc 41이 `0.00784314 ~= 1/127`이라고 쓴 근사가
코드에서 정확한 상수가 되었습니다. 0.6% 차이이고, cos/sin 전 구간에 걸리는
**체계적 편향**입니다.

부수 효과 하나: `zp=128, scale=1/127`이면 `cos=+1`이 `q=255`로 정확히
표현되지만, `scale=2/255`면 표현 범위가 `[-1.00392, +0.99608]`이라
`cos=+1`이 정확히 안 담깁니다. doc 41 §6이 요구한 "uint8 boundaries
(`cos=1 -> 255`)" 테스트가 **스케일에 따라 답이 달라집니다.** 어느
쪽인지부터 확정해야 합니다.

doc 41의 accuracy baseline `cos/sin max absolute error : 0.003937`은
`1/254`, 즉 **자기 자신의 1/127 테이블의 half-step**입니다. 자기 참조라
이 불일치를 잡을 수 없었습니다.

---

## 2. 계약과 설계에서 다시 볼 것

### 2.1 스케일을 어느 텐서에서 읽었는가

doc 41이 값을 뽑은 곳은 `q::mul_op` / `q::Add.tcm`의 "representative
tensors"입니다.

**`q::mul_op`은 커널 타입이지 노드가 아닙니다.** ref_16 §9.1이 GQA
`repeat_kv`로 특정한 것은 노드 `node_expand` / `node_expand_1`이고, 그것이
`q::mul_op` 커널로 lowering된 것입니다. RoPE의 mul도 같은 커널 타입에
매핑됩니다. 그래서 "representative `q::mul_op` 텐서"를 뽑으면 **어느 노드의
것인지 알 수 없고**, 사이클 총량은 GQA가 지배하므로(8,011,003 유닛사이클,
34.7%) 표본이 GQA 쪽으로 기울 가능성이 큽니다.

`Add.tcm [1,8,16,1024]`도 마찬가지로 ref_16 §4.1의 `node_Add_123` --
score/mask 경로 -- 와 같은 커널 타입입니다.

**해결: 커널 타입이 아니라 `args["QNN Op Name"]`(= 원본 ONNX 노드 이름)으로
필터링하세요.** §5.3의 실행 결과로 RoPE 노드가 특정 가능해졌습니다.

그리고 별개로: **cos/sin 테이블의 스케일은 mul의 입출력 텐서가 아니라
cos/sin `$Const` 텐서 자신의 encoding에서 읽어야 합니다.** 지금 코드의
`ROPE_QNN_TABLE_SCALE`은 테이블의 스케일인데 값은 mul 텐서에서 왔습니다.
서로 다른 텐서입니다.

doc 41이 "Do not infer missing values from dtype alone"이라고 옳게 쓰고,
바로 그 문단의 값이 코드에 상수로 들어갔습니다. §1.3과 같은 종류입니다.

### 2.2 `rope_cache_init(theta)`는 default RoPE만 만듭니다

```c
const float inv_freq = powf(theta, -2.0f * (float)k / (float)dim);
```

nntrainer는 `default` / `yarn` / `proportional` 세 가지를 지원하고
(`mha_core.cpp:874-881`), 셋 다 `thetas[half]` + `attention_scaling` 두 개로
환원됩니다. 지금 캐시는:

- `yarn` / `proportional`을 **재현하지 않습니다**
- `attention_scaling`을 **곱하지 않습니다** (`calc_trigonometric_vals_dup`가
  cos/sin에 곱하는 값)
- 따라서 `rope_partial_rotary_factor`를 쓰는 모델(gemma4)과 yarn 모델에서
  **조용히 틀린 테이블**을 만듭니다

**고칠 방법:** `theta` 스칼라 대신 ARM이 이미 계산해 둔 것을 받으세요.

```
AEEResult rope_cache_init(in uint32 n_positions, in sequence<float> thetas,
                          in float attention_scaling, rout uint32 generation);
```

`dim`은 `2 * thetas.len`으로 나옵니다. 세 가지 scaling이 전부 커버되고,
`powf` 루프가 통째로 사라지고, ARM과 DSP가 같은 `thetas`를 쓰므로 불일치
가능성 자체가 없어집니다. doc 41 §3의 cache key 질문도 **대부분 사라집니다**
-- 키는 `(n_positions, thetas 내용의 해시, attention_scaling)`이면 충분하고,
`rope_scaling_type` / YaRN 파라미터 / `partial_rotary_factor`는 전부
`thetas`에 이미 접혀 있습니다.

### 2.3 캐시 메모리의 절반이 죽어 있습니다

`rope_cos_u8` / `rope_sin_u8`은 `rope_cache_init`에서 할당·기록되지만
**어디서도 읽히지 않습니다** -- IDL로 노출되지 않고, 커널은 Q15만 받습니다.
`n_positions=4096, dim=128`에서 `4096 * 64 * 2 = 512 KB`가 DSP 힙에서
낭비됩니다.

읽을 계획이 있으면 지금 노출하고, 없으면 지우세요. doc 41 §8(captured QNN
tensor 비교)이 u8 테이블을 호스트로 꺼내야 한다면 그때 IDL에 붙이는 게
맞습니다.

### 2.4 `powf`가 position 루프 안에 있습니다

`inv_freq`는 `k`에만 의존하는데 `(pos, k)` 이중 루프 안에서 계산됩니다.
`n_positions=4096, half=64`면 `powf` 호출이 **262,144회**, 필요한 건 64회.
§2.2대로 `thetas`를 받으면 이 루프가 사라지므로 같이 해결됩니다.

### 2.5 skel의 선복사가 불필요합니다

`nntr_hvx_rope.c:155-157`이 `memcpy(y, x, expected_x)` 후 커널에 `x`와 `y`를
둘 다 넘깁니다. 커널은 `x`에서 읽고 `y`에 씁니다. §1.1의 memcpy를 지우면
`width % dim == 0`이므로 모든 바이트가 회전으로 덮이고, 이 선복사는
**prefill에서 버퍼 전체를 한 번 더 훑는 순수 낭비**입니다 (Q 2048x1024 = 2 MB).

부수 효과: `x != y`가 항상 참이라 **in-place 경로가 FastRPC로는 한 번도
실행되지 않습니다.** doc 40 §6.2의 in-place 항목은 커널을 직접 부르는
테스트로만 검증할 수 있습니다.

### 2.6 `n_positions <= 4096` / `n_rows <= 4096` 천장이 문서화되지 않았습니다

Qwen3-0.6B의 `max_position_embeddings`는 40,960입니다. 지금 캐시는 position
4096까지만 담습니다. 합리적인 선택이지만 **명시되지 않은 천장**입니다.
`ponytail:` 주석으로 천장과 업그레이드 경로를 적으세요
(`01_working_style.md`).

---

## 3. 테스트 매트릭스가 좁혀졌고, 그 자리에 §1.1이 있습니다

`40_rope_u8_task.md` §6.2는 고정이었고 RULE ZERO §8.2가 변경을 금지했습니다.
실제 구현된 것과의 차이:

| 요구 | 구현됨 | 빠진 것이 숨긴 것 |
| :-- | :-- | :-- |
| `dim` {64, **96**, 128} | {64, 128} | `dim=96 -> half=48 -> 48%32=16`. **tail 경로가 한 번도 안 돌았습니다** |
| `width/dim` {1, **8**, **16**} | {1, 1.5} | **§1.1의 버퍼 오버런.** `width/dim>=2`가 필요한데 전부 1 이하 |
| `n_rows` {1, **7**, **32**, **1024**} | {1, 2, 3} | 대량 row에서의 `s_in[m]` 인덱싱, 그리고 §1.2의 VLA 크기 |
| `zp_in` {0, **128**, **255**} + row별 | {0}, {127}, {17,40,63} | u8 입력에서 `q - zp`가 음수로 크게 가는 경우 |
| broadcast == per-row 동일 결과 | **없음** | broadcast 확장 로직 |
| in-place bitwise 동일 | **없음** | §2.5 때문에 FastRPC로는 검증 불가 |
| amax 비율 / sqrt(2) 상한 출력 | **없음** | |
| §2.1 오차 예산 표 실측 | **없음** | u8->u8 계약의 실제 대가 |

`{3, 96, 64}`는 `width % dim != 0`인 케이스라 그 자체로는 유용하지만,
**`dim=96`을 대신하지 못합니다.** 둘은 다른 축입니다.

doc 41이 인용한 device 결과 `max_lsb=0` 세 줄은 이 좁아진 매트릭스에서 나온
것이라, "커널이 맞다"의 근거로 쓰기엔 약합니다. 매트릭스를 doc 40 §6.2대로
복원한 뒤 다시 재세요.

---

## 4. 이건 아직 HVX 커널이 아닙니다

`rope_block`은 32 lane당:

```
스테이징 스칼라 루프  : ~32 x (u8 로드 2, 정수 뺄셈 2, int->float 변환 4, 저장 4)
벡터 구간            : 언얼라인드 로드 4 + ALU 8 + 저장 2 = 14 명령
출력 스칼라 루프      : ~32 x (덧셈 2, 비교 4, 저장 2)
```

**출력 원소 64개당 스칼라 연산 600여 개**입니다. 벡터 구간 14개는 그 안에
묻힙니다. 명령 수로 유도하면 대략 2 cy/elem 이상이고, ref_16 §3.3이 제시한
elementwise 기준선 **0.07 cy/elem**과 **두 자릿수** 차이입니다. (측정값이
아니라 명령 수에서 유도한 값입니다. 재세요.)

doc 40 §3.2가 지정한 경로가 안 쓰였습니다:

| 지정된 것 | 근거 | 현재 |
| :-- | :-- | :-- |
| `Q6_Wuh_vunpack_Vub` | llama.cpp `hmx-mm-kernels-tiled.h:341` | 스칼라 루프 |
| `Q6_Vh_vpack_VwVw_sat` -> `Q6_Vub_vpack_VhVh_sat` | 우리 `hvx_quant_u8.c:211-213` | 스칼라 clamp 루프 |
| 포화 카운트는 포화 전 `vq` 벡터 비교 | doc 40 §3.2 | 스칼라 `if` |

`hvx_rope_u8.c:20-25`의 주석이 이걸 인정하고 있지만
("The scalar staging is deliberately kept local"), **`ponytail:` 마커가 아니고
천장을 숫자로 적지 않았습니다.** `01_working_style.md`가 요구하는 형태는
"무엇을 포기했고, 그 대가가 얼마이고, 업그레이드 경로가 무엇인지"입니다.

정확도가 먼저이므로 지금 당장 최적화할 필요는 없습니다. 다만 **doc 41 §9가
"performance and memory risks"를 묻는데, 답은 "현재 커널은 목표 대비 두
자릿수 느리고 그건 알려진 상태"** 여야 하고, 지금 문서엔 그 문장이
없습니다.

---

## 5. doc 41의 방향에 대한 분석 -- 순서를 뒤집자

### 5.1 per-op은 양자화 라운드를 늘립니다

RoPE의 ONNX 형태는 `q*cos + rotate_half(q)*sin` -- mul, mul, add. doc 41
§"Target QNN-like path"대로 **각 중간 결과를 u8로 담으면 재양자화가 3번**
입니다. 지금 fused 커널은 1번입니다.

| | rope 내부 라운드 | q가 지고 있는 총 라운드 | 상대 score 오차 |
| :-- | :--: | :--: | --: |
| 현재 fused (`f21d0ad`) | 1 | 2 | ~1.8% |
| per-op mul/mul/add | 3 | 4 | ~2.5% *(유도값)* |

게다가 중간 결과가 `[0,255]`로 **clamp**되므로, QNN이 calibration으로 정한
중간 스케일을 정확히 재현하지 못하면 **포화가 나서 구조적으로 틀립니다.**

즉 per-op은 그 자체로 더 나은 결과를 주지 않습니다. **유일하게 정당한
이유는 "QNN 출력과의 일치를 증명하기 위해"** 이고, 그게 doc 41 §8입니다.

### 5.2 그런데 §8이 §1-7의 필요 여부를 결정합니다

doc 41의 순서는 메타데이터 추출(1) -> CPU ref(2) -> cache key(3) ->
HVX kernel(4) -> IDL(5) -> 테스트(6,7) -> **QNN 캡처 비교(8)** 입니다.

§8이 마지막인데, §8의 결과가 §1-7이 필요했는지를 정합니다:

- QNN의 RoPE **입력과 최종 출력**을 덤프해서
- **지금 있는 fused 커널**에 그 입력과 그 scale/zp를 그대로 먹이고
- 차이를 잽니다

만약 최대 차이가 1-2 LSB 안이면 -- **§1-7 전체가 불필요합니다.** 이미
계약을 만족하고 있고, per-op을 만들면 §5.1대로 오히려 나빠집니다.
차이가 크면, 그 크기가 곧 per-op 작업의 규모를 알려줍니다.

이 실험은 `qnn-net-run --debug` 한 번 + 기존 테스트 하네스 재사용이라
**반나절**입니다. 몇 주를 지울 수 있습니다.

이건 이 프로젝트가 이미 세 번 배운 습관입니다 -- `CLAUDE.md`의
"Measure the breakdown before acting on a hypothesis". FastRPC marshalling
가설이 dominant일 거라 보고 16-32%로 측정됐고, 진짜 비용은 다른 데
있었습니다.

### 5.3 소멸 노드 실행 결과 -- per-op 경계는 살아 있습니다

ref_16 부록의 스니펫을 실제로 돌렸습니다. 소멸 노드 33개:

```
node_Transpose_0  node_view
node_pow_1  node_mean  node_add  node_Sqrt_12  node_rsqrt  node_mul  node_mul_1
node_Transpose_13  node_view_1
node_pow_2  node_mean_1  node_add_1  node_Sqrt_23  node_rsqrt_1  node_mul_2  node_mul_3
node_Transpose_24  node_view_2  node_unsqueeze  node_unsqueeze_1
node_slice_2  node_cat  node_slice_4  node_cat_1
node_unsqueeze_2  node_unsqueeze_3  node__unsafe_view_1
node_Reshape_145  node_Reshape_117  node_transpose_3  node_Transpose_130
```

**해부:**

- `pow_1 / mean / add / Sqrt_12 / rsqrt / mul / mul_1` 7개와
  `pow_2 / mean_1 / add_1 / Sqrt_23 / rsqrt_1 / mul_2 / mul_3` 7개는
  **q_norm / k_norm의 RMSNorm 체인**입니다 -- ref_16 §1.1의
  "`Pow → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul → Mul(weight)`,
  14개 ONNX 노드가 2개 QNN 노드로"와 **정확히 일치**합니다.
  따라서 `node_mul` ~ `node_mul_3`과 `node_add` / `node_add_1`은
  **RMSNorm의 것이지 RoPE의 것이 아닙니다.**
- `slice_2 / cat / slice_4 / cat_1`은 ref_16 §1.4가 말한 `rotate_half`이고,
  `unsqueeze / unsqueeze_1`은 cos/sin의 broadcast 뷰입니다. 전부 0-cost.
- **RoPE의 `mul` / `add`는 이 목록에 없습니다.** 즉 **실행됩니다.**
  ref_16 §7.1의 span 표에 `node_mul_4`가 실제로 올라와 있는 것과 일치합니다
  (77,695 ~ 4,500,706, 27개 노드와 겹침).

**결론: doc 41 §"Target QNN-like path"의 그림이 맞습니다.** `rotate_half`는
공짜 뷰이고, 실행되는 RoPE는 텐서당 `mul(x,cos)` / `mul(rot(x),sin)` / `add`
= **u8 출력 3개**, q와 k 합쳐 muls 4 + adds 2입니다. per-op bit-parity의
대상이 존재합니다.

**따라서 §5.2의 스파이크(S2)로 진행합니다.** 그리고 §2.1의 blocker가
풀립니다 -- 커널 타입이 아니라 `args["QNN Op Name"]`으로 필터링하면
RoPE 노드의 encoding을 정확히 뽑을 수 있습니다.

**주의:** `node_mul_4`가 RoPE의 첫 mul이라는 것은 HF export 순서에서
추론한 것이지 ONNX를 읽어 확인한 것이 아닙니다. 아래로 확정하세요 --
`rotate_half`의 `cat` 출력을 소비하는 노드가 곧 sin 쪽 mul입니다:

```python
import onnx, collections
m = onnx.load('Qwen3-0.6B_prefill.onnx', load_external_data=False)
cons = collections.defaultdict(list)
for n in m.graph.node:
    for i in n.input:
        cons[i].append(n)
for cat in ('node_cat', 'node_cat_1'):
    n = next(x for x in m.graph.node if x.name == cat)
    print(cat, '->', [(c.name, c.op_type) for c in cons[n.output[0]]])
```

숫자 하나 어긋남: 이 스니펫은 소멸 33개를 내놓는데 ref_16 §0은
"실행 33 / 소멸 27"이라고 씁니다. 결론에는 영향이 없지만, ref_16의 그
수치는 재확인이 필요합니다.

---

## 6. 제안하는 순서

```
S1  [완료] RoPE의 mul/add가 실행되는지 확인 -> 실행됨 (§5.3)

S0  §1.1 / §1.2 / §1.3 수정 + doc 40 §6.2 매트릭스 복원 + device 재실행
      -> 지금 "PASSED 5 tests"는 오버런과 좁은 매트릭스 위에 서 있습니다.
         여기부터 다시 시작하지 않으면 그 위의 모든 결론이 흔들립니다.

S1b 노드 이름을 확정하고(§5.3의 두 번째 스니펫) 그 이름으로 필터링해
      RoPE mul/add/Const의 dtype/scale/zp/shape/order를 뽑는다.
      -> doc 42 §7의 blocker 1, 2가 여기서 닫힙니다.
      -> §1.3의 테이블 스케일 실제 값도 여기서 나옵니다 (cos/sin Const에서).

S2  반나절 스파이크: QNN RoPE 입력/최종출력 덤프 -> 현재 fused 커널에 투입
      -> 최대 LSB 차이를 측정 (§5.2)
      -> <= 2 LSB 이면 per-op 불필요. 끝.

S3  (S2가 크게 어긋날 때만) doc 41 §1-7. 단, 스케일은 S1b의 값을 쓸 것,
      그리고 §5.1의 정확도 손해(라운드 1 -> 3)를 PR 본문에 명시할 것.

S4  §2.2 (thetas/attention_scaling을 받는 cache_init) -- S2/S3와 독립.
      yarn/proportional/gemma4가 지금 조용히 틀립니다. per-op 여부와
      무관하게 고쳐야 합니다.

S5  attn_forward 융합 (회전을 hvx_quant_pack_u8_ah의 AH 타일 쓰기 앞으로).
      doc 40 §0이 처음부터 프로덕션 경로로 지목한 것.

S6  §4의 HVX 벡터화 -- 정확도가 고정된 뒤에.
```

**S2를 S3보다 먼저 하는 것이 이 순서의 요점입니다.** per-op 경계가
존재한다는 것(S1)이 per-op을 구현해야 한다는 뜻은 아닙니다 -- §5.1대로
per-op은 재양자화를 1회에서 3회로 늘립니다. S2가 "현재 fused 커널이 이미
QNN 출력과 2 LSB 안"이라고 답하면 S3는 정확도를 **악화시키는** 작업이
됩니다. S2 없이 S3에 착수하지 마세요.

---

## 7. doc 41의 "Review completion criteria"에 대한 답

문서가 리뷰 에이전트에게 요구한 7개 질문 중, 지금 답할 수 있는 것:

| 질문 | 답 |
| :-- | :-- |
| 1. QNN RoPE operator order | **부분 해결 (§5.3).** mul/add는 실행되고 slice/cat/unsqueeze는 0-cost 뷰입니다. 텐서당 mul/mul/add 3단계. 노드 이름 확정만 남음 (S1b) |
| 2. 중간 텐서별 scale/zp | **미해결이지만 경로가 열렸습니다 (S1b).** 지금 코드의 `1/127`은 (a) 관측값 `2/255`와 다르고 (§1.3) (b) 노드가 아니라 커널 타입에서 뽑은 표본입니다 (§2.1) |
| 3. fixed-point vs float 검증 방법 | §5.2의 스파이크(S2)가 선행되어야 질문이 성립 |
| 4. CPU/HVX 비교 지점 | 현재: 최종 출력만. doc 40 §6.2의 lane/tail/in-place 항목이 아직 비어 있음 (§3) |
| 5. 완전한 cache key | **§2.2를 적용하면 대부분 사라집니다** -- `thetas` + `attention_scaling`이 세 scaling type과 partial rotary를 전부 흡수 |
| 6. prefill/decode position 전달 | `position_start`로 이미 있음. 다만 `n_positions <= 4096` 천장 미문서화 (§2.6) |
| 7. 정확도/성능 게이트 | 정확도는 doc 40 §6.2가 이미 고정. 성능 게이트는 없고, 현재 커널은 ref_16 §3.3 기준선 대비 두 자릿수 느립니다 (§4) |

**blocker로 올릴 것: 1, 2.** 나머지는 S0/S2/S4로 진행 가능합니다.
