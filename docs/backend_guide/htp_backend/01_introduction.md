# HTP Backend (HexKL) 소개

Qualcomm Hexagon Tensor Processor (HTP) 백엔드는 Qualcomm HexKL SDK (`libsdkl.so`)를 통해 온디바이스 상에서 트랜스포머 모델의 행렬 곱셈(MatMul/GEMM) 연산을 효율적으로 가속하는 전용 실행 백엔드입니다.

## 1. 연동 배경 및 목표
- **온디바이스 가속**: CPU(NEON) 연산 대비 고대역폭 소모율을 절감하고, Snapdragon 전용 HMX(Hexagon Matrix eXtension) 타일을 활용하여 전력 및 연산 성능을 극대화합니다.
- **유연한 대비**: 고성능 prefill 연산 가속을 지원하고, NPU 구동 실패 시 자동으로 CPU로 무중단 우회하는 Fallback 아키텍처를 목표로 합니다.

## 2. 지원 가능한 연산 스펙
- **FP16 GEMM**: `shgemm_f32f16_f32` (FP32 activation × FP16 weight $\rightarrow$ FP32 C)
- **QINT8 GEMM**: `shgemm_u8i8_i32` (U8 activation × I8 weight $\rightarrow$ I32 accumulation $\rightarrow$ FP32 C)

## 3. 하드웨어 정렬 제약사항
NPU 타일의 병렬 실행 단위 구조적 특성으로 인해 다음과 같은 행렬 차원 정렬 제약이 발생합니다. 다만, 프레임워크 내부 물리 연산 래퍼 계층에서 자동 패딩 기법을 제공하므로 상위 연산에서는 제약이 완화됩니다.

- **M 정렬 (Batch/SeqLen)**: HMX 타일 구조 및 벡터 레지스터 로딩 효율화를 위해 하드웨어 수준에서 정렬을 요구합니다.
  - **FP16 GEMM (`shgemm_f32f16_f32`)**: $M \% 32 == 0$ 정렬 요구
  - **QINT8 GEMM (`shgemm_u8i8_i32`)**: $M \% 64 == 0$ 정렬 요구
  - *M-Padding 지원*: 두 연산 모두 내부 물리 연산 래퍼(`hexkl_mm.cpp`)에서 실제 입력 $M$을 정렬 단위(32 또는 64)로 올림 패딩하여 NPU 버퍼를 할당 및 가공하고, 최종 출력 시 실제 $M$ 영역만 복사해 반환하므로 상위 수준에서는 임의의 $M$ 크기로 동작 가능합니다.
- **N 정렬 (Col/Output Dim)**: $N \% 32 == 0$은 SDKL의 하드웨어 메모리 레이아웃(WH) 정렬을 위한 필수 요구 조건입니다. 이 정렬 조건이 깨질 경우 CPU Fallback을 강제해야 합니다.
- **K 정렬 (Inner Dim)**: SDKL의 HTP 가속 하드웨어 기본 정렬 제약사항입니다.
  - **FP16 GEMM**: $K \% 32 == 0$ 조건을 엄격히 만족해야 합니다.
  - **QINT8 GEMM**: $K \% 64 == 0$ 조건을 엄격히 만족해야 합니다.
