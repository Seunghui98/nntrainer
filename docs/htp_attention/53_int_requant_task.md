# 53 — arXiv 2511.11248: int32 → u8 직접 재양자화, 우리 커널에 되나 (별도 세션 과제, 자체 완결)

상태: **논문은 안 읽었다** — 이 문서를 쓴 컨테이너는 arxiv.org 차단. 아래 §1~§3은 우리 커널 쪽
사실(측정·코드)이고, §4는 논문을 읽은 뒤 채울 빈칸이다. 세션의 첫 일은 논문을 읽고 §4를 채우는 것.

## 0. 한 줄

질문은 "HMX int32 누산기 → f32 → u8인 지금의 에필로그를 int32 → u8로 줄이면 얼마나 빨라지나".
답의 절반은 이미 측정돼 있다: **그 에필로그는 §2.22(51) 이후 HMX 아래에 숨어 있어서, 노출된
몫은 콜당 ~0.35 ms(2.4%)다.** 그러니 이 과제는 "dequant를 빠르게"가 아니라 (a) 숨은 워커 시간이
정말 여유가 있는지 재고, (b) int 경로가 **파이프라인 구조**(행 스캔 배리어, VTCM의 f32 gate_off
448 KB)를 없애 주는지, (c) 정확도(ppl)가 버티는지를 가리는 일이다.

## 1. 지금의 에필로그 — f32 홉이 어디에 있나

`nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c` (MoE 층 콜, prefill 14.4 ms / decode 1.44 ms):

```
gate_up:  HMX u8×i4 → int32 타일(acc_read, VTCM)
          ├ hvx_dq_swiglu_worker (hvx_dequant_i32.c): f32 = (acc − zp·colsum)·act_scale·w_scale + bias   [gate, up 두 타일]
          │                                          silu(gate)·up  f32 (hvx_swiglu_det_sf)  → gate_off f32 [64 × 1792], VTCM 448 KB
          └ moe_dn_worker의 rq 유닛: 행마다 min/max 스캔(hvx_quant_rows_u8_params) → scale, zp
                                   f32 → u8 RNE(hvx_quant_pack_u8_ah_rows)          → mid u8 AH 타일
down:     HMX u8×i4 → int32 → hvx_dequant_acc_tile_to_f32 → res_f32 → scatter-add f32 (라우팅 가중치) → out_c f32
```

- 활성화 양자화는 **행별 동적**(per-row u8, min/max → scale/zp). 가중치는 열별 int4(QS4CX_WH).
  이 레시피가 ppl 62.09를 준다(51 §2.21). 정적/평탄화/SmoothQuant 레시피는 51 §2.11~2.14에서
  전부 졌다(재기준이 Q4_0라 SNR 바닥 ~17 dB; 최종 판정은 ppl).
- 행별 동적이 **파이프라인을 정한다**: 한 행의 scale은 그 행의 1792열이 전부 나와야 나온다 →
  requant는 gate_up 배치 4개가 다 끝난 뒤에만 → 이걸 숨기려고 §2.22에서 down을 한 블록 뒤로 보냈다.
- down 쪽 출력은 f32여야 한다(residual stream이 ARM f32) — int32 → u8은 여기엔 해당 없음.

FC 콜(`hexkl_mm_u8i4_dma.c` `hexkl_mm_u8i4_layer_run`)과 conv 블록(`hexkl_conv_block.c`)도 같은
조각(dequant f32, conv는 phase 2에 requant)이고, 둘 다 풀 에필로그로 숨겨져 있다(51 §2.22~2.27).

## 2. 측정된 것 (51 §2.24, poll 5000, 전부 켬)

| MoE 콜 14.45 ms | us | 비고 |
|---|---:|---|
| mm (HMX 발행) | 9,102 | 타일당 18.5 ns, **천장** |
| acc (acc_read, int32 타일 VTCM 착지) | 2,954 | 타일당 0.37 us, **벤더 함수, 타일당 1회가 최소**(46 §3080) |
| dequant (노출) | 243 | 마지막 에필로그 대기 |
| requant (노출) | 62 | rq를 실은 잡의 대기 |
| scatter (노출) | 44 | |
| **숨은 워커 시간** | **미측정** | MoE 행의 SWIGLU 열은 0. conv 행만 `(hidden)`으로 찍는다(3.0 ms/콜) |

즉 int32→u8이 워커 일을 반으로 줄여도 **노출 0.35 ms 안에서만** 준다 — HMX가 병목인 한 prefill은
안 움직인다. 예외 셋:

1. **acc_read 2.95 ms (21%)**: int32 타일 8 KB/타일. HexKL이 u8/i8 출력 acc_read(고정소수점 스케일 내장)를
   가지고 있다면 4배 작은 읽기 — 이게 있으면 이 과제의 가장 큰 항목. 이 트리의 스텁은 `acc_read_int32`뿐
   (`test/htp/host/stub/hexkl_micro.h`). **SDK의 `hexkl_micro.h`를 열어 `acc_read_*` 변형을 확인하는
   것이 첫 조사**(이 컨테이너엔 SDK 없음).
2. **VTCM**: gate_off f32 448 KB가 u8 112 KB로. 지금 레이아웃 7.03 MB/8.3 — 여유가 생기면 staging
   타일 수나 활성화 슬롯 2벌(51 §2.26이 못 한 것)에 쓸 수 있다.
3. **decode(M=1)**: 콜 1.44 ms 중 dequant 1.1 us, gather 128, acc 259, mm 764 — 에필로그는 0. 해당 없음.

## 3. 정확도 — 이 과제의 진짜 게이트

int32 → u8 직행은 스케일을 **미리** 알아야 한다(고정소수점 곱-시프트). 선택지와 우리 데이터:

| 스케일 | 정확도 근거 | 파이프라인 |
|---|---|---|
| 행별 동적(지금) | ppl 62.09 | 행 스캔 배리어 있음 |
| 행별 동적, int32 도메인에서 스캔 | 수학적으로 지금과 같은 격자(u8 = round((x−min)/scale))를 int32 곱-시프트로 근사 → RNE와 다른 반올림 가능. **ppl로 확인** | 배리어 그대로(스캔이 필요) — 구조 이득 없음, 워커 시간만 감소 |
| 채널별/텐서별 정적(캘리브레이션) | 51 §2.12~2.14의 정적 레시피는 전부 손실. SwiGLU 뒤 활성화는 outlier가 커서 per-row가 필요했다 | 배리어 사라짐, 타일 단위로 u8 즉시 기록 가능 — **구조 이득 최대** |
| 배치(열 묶음)별 동적 | 격자 변경 — 새 레시피, ppl 필요 | 배리어가 배치 단위로 줄어듦 |

논문이 어느 칸인지가 §4다. 정확도 판정은 **ppl(`NNTR_PPL=1`) 하나**: 62.0916 기준, +0.5% 이내면
통과(51 §2.15의 규약). 텍스트 비교 금지(51 §2.9).

## 4. 논문을 읽고 채울 것 (세션 첫 작업)

- [ ] 대상 하드웨어와 누산기 폭: HMX(int32 누산, 64×32 타일)와 같은 모양인가.
- [ ] 스케일의 단위(텐서/채널/토큰/블록)와 결정 시점(정적 캘리브레이션인가, 런타임인가).
- [ ] 비선형(SwiGLU/GELU)이 두 matmul 사이에 있을 때 어떻게 하나 — int8 LUT인가, f32로 갔다 오나.
- [ ] 보고된 이득이 **무엇의** 이득인가: 에필로그 시간인가, 벽시계인가, 메모리인가. 우리는 에필로그가
      이미 숨어 있으므로 벽시계 이득 주장의 전제(에필로그 노출)가 우리와 같은지가 핵심.
- [ ] 정확도 지표와 손실(ppl이면 몇 %).
- [ ] 위 표(§3)의 어느 칸인지. 칸이 정해지면 기대 이득 = (칸의 구조 이득) + (acc_read 변형 유무).

## 5. 만들 순서 (게이트 있음)

| 단계 | 내용 | 게이트 |
|---|---|---|
| 0 | §4 채우기 + SDK `hexkl_micro.h`에서 acc_read 변형 확인 | 이득 산술이 −5 ms 미만이면 **여기서 멈추고 문서에 이유** |
| 1 | MoE 행에 숨은 워커 시간 프로브 추가(conv의 `cb_stage_probe_add` 패턴, `HEXKL_PROBE_SWIGLU` 원자 가산) — 기기 1회 | 워커 시간 vs HMX 발행 시간 비율. 워커가 발행보다 짧으면 "int 경로로 워커를 줄인다"는 효과 0 |
| 2 | (논문 칸에 따라) int32→u8 rq 유닛을 **호스트 스칼라 스텁으로 먼저**(`test/htp/host/`) — 지금 경로와 u8 바이트 비교 | 바이트 차이의 분포(동적-int32 칸이면 RNE 차이 ≤1 LSB인지) |
| 3 | HVX 구현, 기기 ppl + 프로파일 | ppl ≤ 62.4, 노출 dequant/requant 변화, VTCM 절약 |

## 6. 일하는 방식·측정·커밋

52 §6~§7과 동일 — 그 문서의 §6(기기 절차, 디렉터리, 환경변수, 기준값), §7(ponytail 규칙, 호스트 체크,
커밋 형식·명령, 두 브랜치 푸시), §9(함정). 여기 다시 쓰지 않는다. 이 과제는 **커널을 바꾸므로 skel도
재빌드**(`test/htp/build.sh`, 52 §9 마지막 항목). 호스트 체크 `bash test/htp/host/run_host_checks.sh`가
MoE·conv·FC 커널의 비트 동일을 지킨다 — 격자를 바꾸는 변경은 그 참조(`hvx_scalar_stubs.c`의
`quant_row`)도 같이 바뀌어야 하고, 그러면 그 체크는 "구조 검증"으로 돌아간다(51 §2.14의 교훈: 참조가
바뀌면 SNR은 의미를 잃고 ppl만 남는다).

## 7. 다른 세션의 첫 프롬프트 (복사해서 쓰기)

```
nntrainer 저장소, 브랜치 claude/lfm2-moe-ffn-hexkl-2ivn5v (PR nntrainer/nntrainer#4327의 head
claude/htp-lfm2-moe-ffn에도 같이 푸시). 먼저 CLAUDE.md → docs/htp_attention/00_START_HERE.md →
docs/htp_attention/53_int_requant_task.md, 그리고 53이 가리키는 52 §6~§7(측정 절차, 커밋 형식,
ponytail 규칙)을 읽어. 과제: arXiv 2511.11248의 int32→u8 직접 재양자화를 우리 HTP MoE 커널
(hexkl_mm_u8i4_moe.c)에 적용할 수 있는지 분석하고, 된다면 구현. 순서는 53 §5 그대로:
논문을 읽고 53 §4의 빈칸을 먼저 채우고(§3 표의 어느 칸인지), SDK hexkl_micro.h에서 acc_read
변형이 있는지 확인하고, 이득 산술이 −5 ms 미만이면 구현하지 말고 문서에 이유만 남겨.
정확도 게이트는 ppl(NNTR_PPL=1, 기준 62.0916, +0.5% 이내)이고 텍스트 비교는 지표가 아니다.
기기 측정은 전부 내가 한다 — 너는 config·명령을 주고 결과를 받아 문서(53의 다음 절)에 기록해.
가설에 코드 쓰기 전에 분해를 재라 — 51 §2.25~2.27이 왜 그런지의 기록이다.
```
