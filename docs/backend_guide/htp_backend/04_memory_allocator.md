# HTP 메모리 할당 및 물리 버퍼 매핑

Hexagon Tensor Processor는 가상 주소가 아닌 물리 버퍼(ION/RPC 메모리)를 통해 하드웨어 연동 장치와 직접 데이터 DMA 교환을 수행합니다. 이를 위한 메모리 할당 메커니즘을 상세히 다룹니다.

## 1. HtpMemAllocator 역할
HTP 백엔드가 기동 중일 때 메모리 관리 풀(`MemoryPool`)에서 활성화되는 특수 전용 물리 메모리 얼로케이터입니다.
- 헤더: `nntrainer/tensor/htp_backend/htp_mem_allocator.h`
- 구현: `nntrainer/tensor/htp_backend/htp_mem_allocator.cpp`

## 2. 핵심 API 바인딩
- **할당 (`alloc`)**: 내부적으로 Qualcomm SDKL의 전용 물리 메모리 할당 함수인 `sdkl_npu_alloc`을 호출해 버퍼를 생성합니다. 이 영역은 CDSP가 하드웨어 트랜잭션을 바로 보낼 수 있도록 무겁고 정밀하게 캐시 정렬된 물리 메모리입니다.
- **해제 (`free`)**: 소멸 단계에서 `sdkl_npu_free`를 호출하여 물리 DMA 채널 정체를 예방하고 메모리 누수를 완전히 방지합니다.

## 3. 메모리 매핑 흐름
CPU 가상 주소로 생성된 입출력 Activation 데이터 텐서와 가중치(Weight) 텐서는 연산 전 물리 매핑 과정을 거쳐 NPU 코어 내부 타일로 파이프라인 전송됩니다.
- 연산 입출력에 쓰이는 `A`와 `C` 텐서 버퍼 등은 $M$ 차원을 32의 배수 크기(`Mp`)로 라운드업 패딩한 임시 물리 메모리를 잡아 연산을 고속 실행합니다.
- 이 Zero-copy 가속 원리를 통해 커널 연산 시 레이턴시를 최소화합니다.
