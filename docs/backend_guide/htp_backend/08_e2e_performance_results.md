# 실제 단말 E2E 성능 결과

대규모 언어 모델(LLM)의 핵심 트랜스포머 프로젝션 레이어를 대상으로 HTP 백엔드 연동 시 구동되는 Prefill 단계 행렬 연산의 E2E 성능 측정 결과입니다.

**기기:** Galaxy S25 Ultra (SM-S938N, Snapdragon 8 Elite / V79 HTP, ADB `R3CY205ZMND`)  
**빌드:** `armv8_android26/libsdkl.so` + V79 DSP 스켈레톤  
**측정 기준:** 3 warmup + 10 iter 평균 (wall-clock)

---

## 1. FP16 GEMM Prefill 성능 (2026-06-25 실측)

Qwen3-0.6B FC 레이어 5종, **M=16** prefill 기준, NPU `shgemm_f32f16_f32` (transient alloc 경로) vs CPU NEON `shgemm`.

| Layer Name | M | N | K | NPU Kernel (ms) | CPU (ms) | Speedup | Throughput (GFLOPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **f32f16_prefill_q_proj** | 16 | 2048 | 1024 | 20.80 | 19.83 | 0.95x | 3.23 |
| **f32f16_prefill_kv_proj** | 16 | 1024 | 1024 | 10.00 | 9.95 | 1.00x | 3.36 |
| **f32f16_prefill_o_proj** | 16 | 1024 | 2048 | 18.00 | 21.23 | **1.18x** | 3.73 |
| **f32f16_prefill_gate_up_proj** | 16 | 3072 | 1024 | 23.01 | 29.80 | **1.30x** | 4.38 |
| **f32f16_prefill_down_proj** | 16 | 1024 | 3072 | 26.90 | 32.52 | **1.21x** | 3.74 |

**요약:** 5종 중 3종(o_proj, gate_up_proj, down_proj)에서 NPU 우세(1.18-1.30x). q_proj/kv_proj는 M=16 규모에서 CPU와 동등. M이 클수록 NPU 우세 폭 증가 예상.

---

## 2. QINT8 HTP E2E 추론 (2026-06-29 실측)

**모델:** Qwen3-0.6B QINT8, FP32 source -> on-device 양자화  
**경로:** `compute_engine=htp`, QINT8 FC 가중치, 임베딩/LM head FP32

```
prefill: 1 tokens, 1149 ms, 0.870 TPS
generation: 32 tokens, 37999 ms, 0.842 TPS
total: 39313 ms
peak memory: 3252140 KB (~3.1 GB)
```

상태: crash 없이 32 토큰 생성 완료. 출력 품질(반복 패턴)은 QINT8 수치 정확도 개선 후 재검증 예정.

---

## 3. E2E 라우팅 플립 검증 (2026-06-25 실측)

**모델:** Qwen3-0.6B FP32, prompt 5 토큰 -> 32 토큰 생성

| 구간 | CPU 모드 | HTP 모드 | 비고 |
| :--- | :---: | :---: | :--- |
| prefill (5 tokens) | 641 ms / **7.8 TPS** | 3,746 ms / 1.3 TPS | M=5는 FastRPC 왕복 비용이 지배 |
| decode (32 tokens) | **2.96 TPS** | **2.93 TPS** | M=1 -> CPU NEON hsgemv (양 모드 동일 경로) |

**Decode TPS 개선:** 라우팅 플립(decode->CPU 고정) 이전 0.246 TPS -> 이후 2.93 TPS (**11.9x** 향상).

---

## 4. 컬럼 설명

| 컬럼 | 설명 |
| :--- | :--- |
| **M / N / K** | MatMul 차원 (M=Batch/SeqLen, N=가중치 열, K=공통 차원) |
| **NPU Kernel (ms)** | sdkl FastRPC 트랜잭션 완료까지의 wall-clock (transient alloc 포함) |
| **CPU (ms)** | CPU NEON `shgemm` (KleidiAI 기반 단일 스레드) 실행 시간 |
| **Speedup** | CPU 시간 / NPU 시간 |
| **Throughput (GFLOPS)** | FLOPs(2xMxNxK) / (NPU 실행시간 x 10^9) |
