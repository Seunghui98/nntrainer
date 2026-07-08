# HexKL Backend 구성 및 라이프사이클

NPU 세션을 조율하는 `HtpBackend` 싱글톤 클래스의 구동과 라이프사이클을 설명합니다.

## 1. HtpBackend 싱글톤 아키텍처
HTP 백엔드는 무겁고 오버헤드가 큰 초기화 작업을 중복 호출하지 않도록 프로세스 단위의 유일한 싱 글톤 `HtpBackend` 객체를 소유합니다.
- 헤더: `nntrainer/tensor/htp_backend/htp_backend.h`
- 구현: `nntrainer/tensor/htp_backend/htp_backend.cpp`

```cpp
class HtpBackend {
public:
  static HtpBackend &global();
  bool enabled() const { return enabled_; }
  int domain() const { return domain_; }
};
```

## 2. 초기화 및 해제 라이프사이클
- **지연 초기화 (Lazy Initialization)**: 프로그램이 처음 시작될 때가 아닌, HTP 컨텍스트나 커널(`get_htp_ops()`)에 최초 접근 시점에 싱글톤 내부 생성자에서 `sdkl_npu_initialize(domain, ...)`가 1회 실행됩니다.
- **버전 질의**: 초기화 성공 시 `sdkl_npu_get_version`을 호출해 CDSP 펌웨어 사양을 로그에 출력 합니다.
- **자원 정리**: 프로그램 프로세스가 정상 종료되는 소멸자 단계에서 `sdkl_npu_finalize(domain)` 를 안전하게 기인하여 점유 자원을 반환합니다.
  - `HtpBackend`는 함수-로컬 static이라 namespace-scope 정적(NPU 스크래치/캐시)보다 **먼저** 소멸된다. 과거에는 그 정적들의 소멸자가 이미 finalize된 NPU에 `sdkl_npu_free`를 호출해 종료 시 `Err=1`이 3회 출력되었다.
  - 현재는 `HtpBackend`가 `sdkl_npu_finalize(domain_)` **직전**에 원자 플래그 `g_npu_alive`를 `false`로 내린다(`bool npuAlive()`로 조회). 모든 teardown free는 `npuFreeIfAlive()`로 감싸져 NPU가 죽은 뒤에는 free를 건너뛴다. finalize가 메모리를 회수하므로 이는 누수가 아니라 안전한 skip이다. (커밋 `62a11ef3`)

## 3. 무중단 CPU Fallback 설계
HTP 하드웨어가 없는 테스트 PC 환경이거나 단말의 CDSP 스켈레톤(`/data/local/tmp/libhexkl_skel.so`) 경로 오류 등으로 인해 초기화가 실패할 경우를 완벽하게 방지합니다.
- `sdkl_npu_initialize` 결과 코드가 0이 아닐 경우 로그 경고(`ml_logw`) 후 내부 `enabled_` 상태 를 `false`로 남깁니다.
- 이후 실행되는 `HtpComputeOps`의 지원 판별 함수 `supports_shgemm()`은 싱글톤의 `enabled()` 값 을 추적하여 무조건 `false`를 리턴하며, 연산 계층(`float_tensor.cpp` 등)은 이 신호를 확인하여 CPU NEON 연산으로 안전하고 투명하게 제어를 전환합니다.
