<!-- SPDX-License-Identifier: Apache-2.0 -->

# 43 -- u8 RoPE 커널 수정 태스크 (S0), self-contained

`42_rope_u8_review_and_next.md`의 S0입니다. 리뷰에서 나온 결함을 고치고
`40_rope_u8_task.md` §6.2의 테스트 매트릭스를 복원해서, 그 위에 올라갈
모든 결론이 흔들리지 않게 만드는 것이 목적입니다.

§8에 구현 에이전트에게 붙여넣을 프롬프트가 있습니다.

---

## 0. "구현이 잘못됐나?" -- 층을 나눠서

| 층 | 판정 |
| :-- | :-- |
| **회전 산술** | **맞습니다.** `40_rope_u8_task.md` §1.2 그대로이고 호스트 레퍼런스와 일치합니다 |
| **u8-in/u8-out 계약** | **맞습니다.** scale/zp를 받고 `n_saturated`를 돌려주는 형태까지 |
| **메모리 안전성** | **틀렸습니다.** 버퍼 밖 쓰기 1개, 스택 오버플로 위험 1개 (§2.1, §2.2) |
| **테이블 상수** | **틀렸습니다.** 근사값이 정확한 상수가 됐습니다 (§2.3) |
| **rope scaling 지원 범위** | **불완전합니다.** default만. yarn/proportional/gemma4가 조용히 틀립니다 (§2.4) |
| **테스트 커버리지** | **좁혀졌습니다.** 그리고 빠진 축이 위 결함 자리입니다 (§5) |
| **fused vs per-op 구조** | **아직 판정 대상이 아닙니다.** doc 42 S2가 정합니다. 이 태스크에서 건드리지 마세요 |
| **HVX 벡터화** | 틀린 게 아니라 느립니다. 정확도 고정 후, 별도 태스크 (§1 범위 밖) |

즉 **다시 쓰는 게 아니라 고치는 것**입니다. 예상 diff는 200줄 안쪽입니다.

---

## 1. 범위

**하는 것:** §2의 결함 6개 수정, §3의 IDL 변경, §4~5의 테스트 복원과 신규
가드 테스트, device 재실행.

**범위 밖 -- 손대지 마세요:**

| 항목 | 이유 |
| :-- | :-- |
| per-op mul/add 양자화 (doc 41) | doc 42 S2가 필요 여부를 정합니다. S2 전에 착수 금지 |
| `rope_block`의 스칼라 스테이징을 벡터화 | 정확도 고정이 먼저. 별도 태스크 (doc 42 §4) |
| `attn_forward` / `attn_kv_append` 융합 | 다음 태스크 |
| K 경로 | K는 u8이 아닙니다 (`40_rope_u8_task.md` §1.5) |

---

## 2. 수정 목록

### 2.1 `hvx_rope_u8.c` -- 버퍼 밖 쓰기 제거 [필수]

**현재 (`:91-99`)**

```c
for (uint32_t w = 0; w + dim <= width; w += dim) {
  for (uint32_t offset = 0; offset < half; offset += ROPE_LANES) {
    rope_block(row + w, out + w, offset, half, cq, sq, zp_in[m], vinv,
               zp_out, n_saturated);
  }
  if (dim < width) {
    memcpy(out + w + dim, row + w + dim, width - dim);   /* <-- 삭제 */
  }
}
```

`out`은 row의 base이지 `out + w`가 아닙니다. 마지막 반복
(`w = width - dim`)의 목적지는 `out + width` -- row의 끝 -- 이고 길이는
`width - dim`입니다. `width=2048, dim=128`에서 마지막 row면 FastRPC 버퍼
밖으로 **1,920 바이트**입니다. 각 복사를 다음 반복이 회전으로 덮기 때문에
**출력은 맞고, 출력 비교 테스트로는 안 잡힙니다.**

**고침:** `if` 블록을 지우고, 꼬리는 루프 **밖에서 한 번만** 처리합니다.

```c
  const uint32_t rotated = width - (width % dim);
  for (uint32_t w = 0; w < rotated; w += dim) {
    for (uint32_t offset = 0; offset < half; offset += ROPE_LANES) {
      rope_block(row + w, out + w, offset, half, cq, sq, zp, vinv,
                 zp_out, n_saturated);
    }
  }
  /* Channels past the last whole `dim` segment are not rotated. In practice
     width % dim == 0 (width = n_heads * head_dim), and partial rotary is
     handled by zeroed thetas, not here -- see 40_rope_u8_task.md §1.4. This
     is the contract's tail, not a partial-rotary path. */
  if (out != row && width % dim) {
    memcpy(out + rotated, row + rotated, width % dim);
  }
```

`out != row` 가드가 필요합니다 -- 헤더가 `y may alias x`를 약속하고 있고
`memcpy(p, p, n)`은 UB입니다.

`memcpy`가 이것뿐이면 `#include <string.h>`는 남겨 두세요 (조건부로 여전히
씁니다).

### 2.2 `nntr_hvx_rope.c` -- VLA 제거 [필수]

**현재 (`:148-153`, `:186-191`)**

```c
float row_scale[n_rows];
int32 row_zp[n_rows];
```

`n_rows <= 4096`이므로 **16 KB + 16 KB = 32 KB**입니다. 이 트리의 QuRT
워커 스택은 16 KB이고(`hvx_worker_pool.c:38`) FastRPC 서비스 스레드도 같은
자릿수입니다. 넘치면 예외가 아니라 조용한 손상입니다.

**고침:** 커널에 stride를 넘깁니다. 배열도 복사도 없습니다.

```c
/* hvx_rope_u8.h */
void hvx_rope_u8_rows(const uint8_t *x, uint8_t *y,
                      uint32_t m_first, uint32_t m_last,
                      uint32_t width, uint32_t dim,
                      const float *s_in, uint32_t s_stride,
                      const int32_t *zp_in, uint32_t zp_stride,
                      const int16_t *cos_q15, const int16_t *sin_q15,
                      float s_out, int32_t zp_out, uint32_t *n_saturated);
/* s_stride / zp_stride: 0 = one value broadcast to every row, 1 = per row.
   The IDL lets the two lengths differ independently, so they get one
   stride each rather than sharing one. */
```

커널 안: `s_in[(size_t)m * s_stride]`, `zp_in[(size_t)m * zp_stride]`.
skel: `s_inLen == 1 ? 0u : 1u`.

### 2.3 + 2.4 `rope_cache_init` -- 시그니처 교체 [필수]

결함 4개가 한 번에 닫힙니다.

**현재의 문제**

| # | 문제 | 위치 |
| :-- | :-- | :-- |
| a | 테이블 스케일이 `1.0f/127.0f`로 하드코딩. 트레이스 관측값은 `0.00784314 = 2/255` | `:21` |
| b | `inv_freq = powf(theta, -2k/dim)` -- **default rope만**. `yarn` / `proportional`을 재현하지 않음 | `:90` |
| c | `attention_scaling`을 cos/sin에 곱하지 않음 (`calc_trigonometric_vals_dup`는 곱함) | `:93-94` |
| d | `powf`가 `(pos, k)` 이중 루프 안. `4096 x 64 = 262,144`회, 필요한 건 64회 | `:90` |

(b)+(c) 때문에 `rope_partial_rotary_factor`를 쓰는 모델(gemma4)과 yarn
모델에서 **조용히 틀린 테이블**이 나옵니다.

**고침: ARM이 이미 계산해 둔 것을 받으세요.** `mha_core.cpp:871`의
`precompute_freqs`가 `thetas`와 `attention_scaling`을 만들고, 세 가지
scaling type이 전부 그 둘로 환원됩니다.

```
AEEResult rope_cache_init(in uint32 n_positions,
                          in sequence<float> thetas,
                          in float attention_scaling,
                          in float table_scale, in int32 table_zp,
                          rout uint32 generation);
```

- `dim = 2 * thetas.len` -- 별도 파라미터 불필요
- DSP는 `cos(pos * thetas[k]) * attention_scaling`만 계산 -- `powf` 루프 소멸
- `table_scale` / `table_zp`는 **파라미터**입니다. 어떤 상수도 하드코딩하지
  마세요. 실제 값은 doc 42 S1b가 QNN의 cos/sin `$Const` 텐서 encoding에서
  뽑습니다. **그때까지 테스트는 호출자가 주는 값으로 돌립니다.**
- 캐시 키 = `(n_positions, thetas 내용, attention_scaling, table_scale, table_zp)`.
  `thetas`는 64개 float이므로 복사해 두고 `memcmp`로 비교하면 됩니다.
  해시 불필요. **doc 41 §3이 묻는 cache key 질문이 여기서 닫힙니다** --
  `rope_scaling_type` / YaRN 파라미터 / `partial_rotary_factor`는 전부
  `thetas`에 이미 접혀 있습니다.

DSP libm의 `cosf`와 ARM의 것이 마지막 ulp에서 다를 수 있지만, u8 양자화
step(`~7.8e-3`)에 비해 무의미합니다. 주석으로 한 줄 남기세요.

### 2.5 `nntr_hvx_rope.c` -- 죽은 u8 테이블 삭제 [권장]

`rope_cos_u8` / `rope_sin_u8`은 할당·기록되지만 **어디서도 읽히지
않습니다.** IDL로 노출되지 않고 커널은 Q15만 받습니다.
`n_positions=4096, dim=128`에서 512 KB가 DSP 힙에서 낭비됩니다.

**고침:** 두 필드를 `nntr_hvx_session.h`에서 지우고, `rope_cache_free` /
`nntr_hvx_close`(`hvx_add_f32.c:159-162`)의 `free`도 같이 지웁니다.
호스트가 테이블을 비교해야 하면 `thetas`로 직접 계산할 수 있으므로 DSP
사본이 필요 없습니다.

### 2.6 `nntr_hvx_rope.c` -- 선복사 삭제 [권장]

`:155-157`, `:193-195`의 `memcpy(y, x, expected_x)`는 §2.1 수정 후
불필요합니다 (`width % dim == 0`이므로 모든 바이트가 회전으로 덮임).
prefill Q(2048x1024)에서 **2 MB를 한 번 더 훑는 순수 낭비**입니다.

**단, §2.1의 꼬리 처리가 `out != row`일 때만 도는 것에 의존합니다.**
선복사를 지우면 `x != y`인 호출에서 꼬리가 커널의 post-loop memcpy로
채워집니다 -- §2.1의 코드가 그렇게 되어 있는지 확인하세요.

### 2.7 `ponytail:` 주석 2개 추가 [필수]

`01_working_style.md`가 요구하는 형태(포기한 것, 대가, 업그레이드 경로)로:

```c
/* hvx_rope_u8.c, rope_block 위 */
/* ponytail: the u8 widen and the u8 pack are scalar loops around a
   14-instruction vector core -- roughly 600 scalar ops per 64 outputs,
   which is two orders of magnitude off ref_16 §3.3's 0.07 cy/elem
   baseline for elementwise kernels. Accuracy is fixed first on purpose.
   Upgrade path: Q6_Wuh_vunpack_Vub in, Q6_Vh_vpack_VwVw_sat ->
   Q6_Vub_vpack_VhVh_sat out (the chain hvx_quant_u8.c:211-213 already
   uses), and a vector compare for the saturation count. */

/* nntr_hvx_rope.c, rope_cache_init 위 */
/* ponytail: n_positions is capped at 4096 while Qwen3-0.6B's
   max_position_embeddings is 40,960, because the table is fully resident
   (n_positions * dim/2 * 2 B). Upgrade path: cache a sliding window keyed
   on position_start, or drop the cache and pass the table per call --
   decode needs one row. */
```

`hvx_rope_u8.c:20-25`의 기존 주석은 이걸로 대체하세요 -- 현재 문구는 사실을
말하지만 대가를 숫자로 적지 않았습니다.

---

## 3. IDL 변경 요약

| 엔트리 | 변경 |
| :-- | :-- |
| `rope_u8` | **없음.** stride는 커널 내부 파라미터이고 skel이 계산합니다 |
| `rope_cache_init` | **교체** -- §2.3의 새 시그니처 |
| `rope_u8_cached` | **없음** |
| `rope_cache_clear` | **없음** |

`qaic`가 스칼라 out에 `out`을 거부하고 `rout`만 받는 것에 주의하세요
(이미 물린 함정).

---

## 4. 신규 테스트 -- §2.1을 실제로 잡는 것

출력 비교로는 §2.1이 안 잡히므로 **가드 바이트 테스트**가 필요합니다.

```cpp
TEST_F(HvxRope, DoesNotWritePastTheOutputBuffer) {
  /* width/dim >= 2 and width % dim == 0 -- the shape the previous matrix
     never had, and the one the tail memcpy overran on. */
  const uint32_t rows = 2, width = 512, dim = 64, half = dim / 2u;
  constexpr uint8_t kGuard = 0xA5;
  const size_t payload = (size_t)rows * width;
  const size_t guard = 2048;

  std::vector<uint8_t> x(payload), y(payload + guard, kGuard);
  /* ... fill x, scale, zp, cos, sin ... */

  ASSERT_EQ(nntr_hvx_rope_u8(handle_, rows, width, dim, x.data(),
                             (int)payload, ..., y.data(), (int)payload,
                             &sat), AEE_SUCCESS);

  for (size_t i = payload; i < y.size(); ++i) {
    ASSERT_EQ(y[i], kGuard) << "wrote " << (i - payload) << " bytes past the end";
  }
}
```

`yLen`으로는 `payload`를 넘기되 버퍼는 더 크게 잡는 것이 핵심입니다.
FastRPC가 `yLen` 바이트만 매핑하면 커널의 오버런이 매핑 밖으로 나가
fault가 날 수도 있는데, **그것도 통과가 아니라 실패로 보이면 됩니다.**

---

## 5. 테스트 매트릭스 복원

`40_rope_u8_task.md` §6.2가 고정값이고 RULE ZERO §8.2가 변경을 금지했는데
좁혀져 있습니다. 복원할 것:

| 축 | 현재 | 복원 |
| :-- | :-- | :-- |
| `dim` | {64, 128} | **{64, 96, 128}** -- 96은 `half=48`, `48%32=16` 꼬리. **지금 tail 경로가 한 번도 안 돌았습니다** |
| `width/dim` | {1, 1.5} | **{1, 8, 16}** -- §2.1이 사는 자리 |
| `n_rows` | {1, 2, 3} | **{1, 7, 32, 1024}** |
| `zp_in` | {0}, {127}, {17,40,63} | **{0, 128, 255} + row별로 다른 값** |
| broadcast == per-row | 없음 | 같은 값을 길이 1과 길이 `n_rows`로 주면 **결과 동일** |
| in-place | 없음 | 커널을 직접 부르는 경로에서 `y==x`가 별도 버퍼 결과와 **bitwise 동일** (FastRPC는 항상 별도 버퍼라 skel로는 검증 불가) |
| amax 비율 | 없음 | 회전 후/전 `amax` 비율 출력, `sqrt(2)` 상한 안인지 |
| 오차 예산 | 없음 | `40_rope_u8_task.md` §2.1 표를 실측으로: 요소 RMS, 상대 score 오차 |
| saturation | `EXPECT_GT(sat,0)`만 | 의도적으로 작은 `s_out`에서 **레퍼런스와 정확히 일치** |

`{3, 96, 64}`는 `width % dim != 0` 케이스라 유용하지만 **`dim=96`의 대체가
아닙니다.** 다른 축입니다. 둘 다 넣으세요.

`rope_cache_init` 시그니처가 바뀌므로 `GeneratesQnnUint8CacheAndReusesIt`의
호스트측 테이블 생성도 `thetas` + `attention_scaling` + `table_scale` /
`table_zp`로 맞춰야 합니다.

---

## 6. 검증 순서

```bash
# 1. 호스트 -- DSP 없이 잡히는 것부터
meson build -Denable-transformer=true && ninja -C build
cd build && meson test unittest_mha_htp_host_model --print-errorlogs

# 2. skel (-Werror에서 warning 0)
cd test/htp && bash build.sh

# 3+4. ARM 빌드 + push + run
bash test/htp/run_u8i4_layer_on_device.sh
# RoPE만:
adb shell "cd /data/local/tmp/htp_u8i4_layer_test && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_rope"
```

**게이트:**

1. `DoesNotWritePastTheOutputBuffer` 통과 (§4)
2. 복원된 매트릭스 전부 통과, `max_lsb <= 1` (§5)
3. `n_saturated`가 모든 케이스에서 레퍼런스와 **정확히 일치**
4. §5의 출력 3종(amax 비율, 오차 예산 표, mismatch 비율)이 로그에 찍힘

**수정 전후를 같은 매트릭스로 돌려서 비교하세요.** 지금 `PASSED 5 tests`는
좁은 매트릭스에서 나온 것이라 before로 쓸 수 없습니다 -- 수정 전 코드에
복원된 매트릭스를 먼저 물려서 **어느 케이스가 실제로 깨지는지** 기록하고,
그 다음 고치세요. 그게 §2.1이 이론이 아니라 실측이 되는 유일한 방법입니다.

Hexagon SDK나 디바이스가 없으면 1번까지만 하고 **"device는 못 돌렸다"를
보고서 첫 줄에 쓰세요.**

---

## 7. RULE ZERO

1. **doc 42 S2 전에 per-op 구현을 시작하지 마세요.** 경계가 존재한다는
   것(doc 42 §5.3)이 재현해야 한다는 뜻이 아닙니다 -- per-op은 재양자화를
   1회에서 3회로 늘립니다.
2. **어떤 양자화 상수도 하드코딩하지 마세요** (§2.3). `table_scale` /
   `table_zp` / `s_out` / `zp_out`은 전부 파라미터입니다. 실제 값은
   doc 42 S1b가 QNN 메타데이터에서 뽑습니다.
3. `40_rope_u8_task.md` §6.2의 매트릭스와 허용오차를 **줄이지 마세요.**
   지난번에 줄인 두 축이 정확히 결함 자리였습니다.
4. 실패한 테스트를 bound를 낮추거나 케이스를 빼서 고치지 마세요. 멈추고
   전체 출력과 함께 보고하세요.
5. HVX/HexKL/QuRT API를 발명하지 마세요. 없으면 멈추고 보고하세요.
6. `nntrainer/tensor/htp_backend/` 아래에서 이 태스크가 건드리는 것은
   `hvx_rope_u8.{c,h}` **뿐**입니다. 다른 파일은 device-verified입니다.
7. `git commit -s`, `Co-authored-by:` 트레일러, `[<component>] <subject>`,
   바뀐 줄에 `clang-format-14`.
8. 실행한 모든 명령의 **완전한 출력**을 붙이세요. 요약 금지.

---

## 8. 붙여넣을 프롬프트

```
You are fixing defects in nntrainer's Hexagon HVX uint8 RoPE kernel.

BEFORE WRITING ANY CODE:
1. Read docs/htp_attention/01_working_style.md.
2. Read docs/htp_attention/42_rope_u8_review_and_next.md sections 1-3.
3. Read docs/htp_attention/43_rope_u8_fix_task.md in full. That is your task.
4. Then, in your first reply and before writing code:
   a) explain in your own words why the memcpy in hvx_rope_u8.c:96-98
      writes past the end of the buffer, and why comparing outputs against
      the reference cannot detect it;
   b) state why rope_cache_init must take thetas rather than a scalar
      theta;
   c) list the files you will modify, and stop for one round if that list
      includes anything section 1 or RULE ZERO told you not to touch.

Work in this order:
  1. Run the RESTORED test matrix (section 5) against the CURRENT,
     unfixed code and record which cases fail. That is the before.
  2. Apply the fixes in section 2.
  3. Re-run. Paste both runs in full.

RULE ZERO is section 7. It is not negotiable. In particular: do not start
per-op mul/add quantization, and do not hardcode any quantization
constant -- table_scale and table_zp are parameters whose real values
come from a separate QNN metadata extraction task.

If you do not have the Hexagon SDK or a device, do the host test only and
say plainly in the first line of your report that the device run did not
happen. Do not report a device result you did not observe.
```
