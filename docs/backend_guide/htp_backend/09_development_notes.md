# HTP 백엔드 개발 현황 (Development Notes)

이 문서는 HTP(HexKL) 백엔드의 개발 현황을 추적한다. 각 주제의 상세는 아래 가이드 문서를 참조:
[01 소개](01_introduction.md) · [02 빌드](02_build_and_env.md) · [03 라이프사이클](03_backend_session.md) ·
[04 메모리](04_memory_allocator.md) · [05 커널/캐시](05_operators_and_cache.md) ·
[06 유닛테스트](06_unittest_guide.md) · [07 테스트 결과](07_unittest_results.md) ·
[08 E2E 성능](08_e2e_performance_results.md)

## 1. 상태 대시보드

| 기능/컴포넌트 | 상태 | 미해결 이슈 | 다음 단계 |
| :--- | :---: | :--- | :--- |
| FP16 GEMM prefill (NPU shgemm) | ✅ | — | — |
| QINT8 GEMM (u8i8_i32) | 🚧 | 출력 품질(수치 정확도) 재검증 필요 | 양자화 정확도 개선 후 재측정 |
| Decode 라우팅 (M=1 → CPU 고정) | ✅ | — | — |
| Pin-once WH residency (48 MB 캡) | ✅ | 실효 예산 ~48 MB 한계 | 오프라인 bake로 대체됨 |
| 오프라인 WH bake (WHF1 트레일러) | ✅ | peak RSS +1.6 GiB 트레이드오프 | 메모리 타이트 기기 배포 판단 |
| HtpMemAllocator | ⏸️ 비활성 | DMA 풀 고갈로 미등록 | per-call staging 유지 |
| 진단 토글 (NNTR_HTP_*) | ✅ | — | — |
| armv9 libsdkl.so SIGILL | ✅ 해결 | — | — |
| 최신 성능/정확성 | 🚧 측정중 | — | Phase D에서 07·08 링크로 반영 |

(범례: ✅ 완료 / 🚧 진행중 / 📋 계획 / ⏸️ 비활성 / ⚠️ 우회)

## 2. 알려진 이슈 / 트레이드오프

- **WH-baked bin 메모리 비용**: integrated bin은 RM 대비 파일 +~880 MB, 실행 peak RSS +~1.6 GiB. prefill 대폭 개선과의 트레이드오프 (08 참조).
- **QINT8 출력 품질**: 32토큰 생성은 crash 없이 완료되나 수치 정확도 재검증 예정.
- **E2E single-run 분산**: run-to-run ±300–400 ms. 효과 크기가 분산보다 클 때만 유의미.

## 3. 타임라인 (devlog)

| 날짜/커밋 | 내용 |
| :--- | :--- |
| `d93ece76` | 백엔드 skeleton + HtpContext/ComputeOps 인프라 |
| `93f281fc` | shgemm_f32f16_f32 구현 + prefill/decode 라우팅 |
| `aaa544ee` | QINT8 u8i8 경로 + CausalLM 벤치마크 |
| `ff85baf3` | 영구 WH residency 캐시 + prefill warmup |
| `9a054dee` | HtpMemAllocator 제거; NDK r26d 빌드 수정 |
| `b765c7bb` | evicting 캐시 → pin-once residency |
| `d49395f8` | pre-baked WH 레지스트리 + 스크래치 재사용 |
| `daea0bbe` | 오프라인 WH bake: byte-identity 테스트 + 트레일러 writer |
| `98db4e57` | 로드 시 WH 트레일러 로드; bake-gate dtype/예외 안전 수정 |
| `537693ca` | nntr_quantize에 FP16_WH bake 연결; INFERENCE 등록 수정; E2E speedup 기록 |
| 2026-07-07 `62a11ef3` | finalize 후 sdkl_npu_free skip (npuAlive 가드, Err=1 수정) |
| 2026-07-07 `88779af0` | consumer 빌드 수정: ENABLE_FP16=0 + wh_trailer.h install |
| 2026-07-07 `76feb49b` | pre-baked WH 진단 토글 추가 |
| 2026-07-07 `8c1c1fa5` | **weight-layout 버그 수정** — [K,N]/[N,K] 오라벨로 NPU 출력 garbage였던 것 수정; coherent 출력 확인 |
