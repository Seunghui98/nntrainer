# 57. gemma-4-26B-A4B: int2 expert 가중치 (flash → arena int2 → VTCM에서 int4로 확장)

상태: **구현·호스트 검사 완료, 기기 gtest 통과(§5.1), e2e 성능 미측정.** 브랜치 `claude/epic-hopper-occf31`, 커밋 a77ccbe·016f9d4·4aaa7c5·b816b18 + 이 문서.

## 1. 과제와 범위

- **expert 가중치만** int2로 둔다. 각 expert의 gate_up과 down이 대상이다. attention projection, dense MLP, router, embedding/lm_head는 지금 그대로다(Q4_0 또는 int4 WH).
- flash에서 int2로 읽고, arena에도 int2로 두고, DMA로 VTCM까지 int2로 옮긴 뒤 VTCM 안에서 int4로 확장한다. HMX는 지금과 같은 int4 WH 타일을 읽는다.
- 확장은 파이프라인 안에 숨긴다. 노출되는 몫은 프로파일 칸으로 잰다(§2.4).
- 측정은 Gemma에서 e2e로, read-ahead 끔/켬 두 모드로 한다. 가중치는 더미다(§3.2).

## 2. 설계

### 2.1 형식 WH2

- WH 레이아웃(htp_wh_layout.h)의 각 니블을 2비트 코드로 줄인 것이다. 바이트 수는 정확히 절반이다.
- WH2 바이트 j는 WH 바이트 2j, 2j+1을 담는다. 슬롯 4j..4j+3이 2비트씩 들어가고 슬롯 4j가 비트 0-1이다.
- 코드 c는 int4 값 c−2다. 즉 int2 범위는 [−2, 1]이다.
- 바이트 단위 압축이라 가중치 안의 모든 오프셋이 절반이 된다. 청크·k-tile 단위로 자유롭게 쪼개 확장할 수 있다.
- 정의: `nntrainer::wh2Bytes`, `wh2FromWh`. 확장: `hvx_expand_wh2`(벡터 1개 입력, 2개 출력). 스칼라 쌍둥이는 호스트 스텁에 있다.

### 2.2 파일과 arena

- 파일은 **int4 레이아웃을 그대로 유지**한다. WH2 가중치는 니블 구간의 앞 절반이고, 뒤 절반은 읽지 않는다. scale과 colsum은 원래 자리(off + whBytes)에 있다. 그래서 로더 오프셋이 하나도 바뀌지 않는다.
  - ponytail: 파일 크기는 int4와 같다. 압축 파일을 원하면 양자화기가 따로 써야 하고 expert 오프셋 공유가 깨진다.
- `NNTR_MOE_EXPERT_BITS=2`면 호스트는 니블 구간의 앞 절반만 pread하고, arena 칸 stride도 절반으로 잡는다.
- 첫 expert를 놓기 전에 DSP에 `set_expert_bits(2)`를 한 번 보낸다. 이후 swap은 arena에서 WH2로 읽고, scale·colsum은 WH2 바로 뒤에서 읽는다.

| | int4 (WH) | int2 (WH2) |
|---|---|---|
| expert 칸 (gate_up+down, 4 KB 정렬) | 3,010,560 B (2.871 MiB) | 1,523,712 B (1.453 MiB) |
| miss 1회 flash 읽기 | 3.01 MB | 1.52 MB |
| expert 1개 DMA (DDR→VTCM) | 2.84 MiB | 1.42 MiB |
| C=16 arena | 1408 MiB | 704 MiB |
| C=32 arena | — | 1408 MiB (int4 C=16과 같음) |
| C=16 cold prefill 바닥 (55 §4 방식, 3.0 GB/s) | 10.12 GB → 3.37 s | 5.12 GB → 1.71 s |

표는 전부 산술이다. **기기 미측정.** int4 C=16 arena 1408 MiB는 55 §4 값과 같아 칸·청크 산식은 맞다.

### 2.3 DSP 파이프라인 (`hexkl_mm_u8i4_moe.c`)

- VTCM: WH2일 때 가중치마다 staging 버퍼 2개를 arena 위쪽에서 떼어 낸다. Gemma 형상은 2.84 MiB다.
  - 나머지 레이아웃 4.45 MiB와 합쳐 7.29 MiB로 8.11 MiB 안에 들어간다(호스트 검사 출력).
  - LFM2는 staging이 5.25 MiB라 안 들어간다. 호출이 `AEE_ENOMEMORY`로 거부한다.
- expert k+1의 int2 바이트는 **한 expert 앞서** staging `(k+1)&1`로 DMA한다.
- int4 경로가 가중치를 DMA하던 바로 그 자리, 즉 VTCM의 w_gu/w_dn이 비는 순간에 세 가지를 한다(`moe_wh2_step`).
  1. staging k&1의 DMA를 기다린다. 한 expert 전에 넣었으므로 이미 도착해 있다.
  2. 확장을 **워커 풀의 background lane**에 job으로 넣는다. 단위는 청크 × k-tile 8분할이고 청크 순서다.
  3. 다음 expert의 int2를 다른 staging에 push한다.
- matmul 루프는 int4 경로가 DMA 청크 c를 기다리던 곳에서 "확장 job의 청크 c까지"를 기다린다(`hvx_worker_pool_wait_bg`). gate_up의 gate/up 쌍 청크 구조도 그대로다.
- 왜 background lane인가: 이 스레드에서 확장하면 HMX issue 사이에 끼어 전부 노출된다. background lane에 넣으면 워커가 epilogue 사이 유휴 시간(47 §14)에, 앞 블록의 matmul 아래에서 처리한다.
- background lane은 FIFO다. 그래서 expert 0의 gate_up 확장은 **pack job보다 먼저** 넣는다. pack 뒤에 넣으면 호출 전체의 pack이 끝나야 시작된다.
- 확장 job 구조체는 expert마다 따로 둔다(`jobs[1+2i]`, `jobs[2+2i]`, WH2에선 tail 경로가 꺼져 그 칸이 빈다). 같은 구조체를 재제출하면 pool 링에 남은 옛 칸이 그 job을 일찍 열 수 있다.
- tail 경로(HVX GEMM이 wh_bytes를 직접 읽음)는 WH2에서 끈다. 기기 기본값 `MOE_TAIL_MAX_ROWS=0`이라 기기에서는 원래 꺼져 있다.
- 한 호출 안에서 WH2와 WH expert를 섞으면 `AEE_EBADPARM`이다.

### 2.4 무엇이 숨고 무엇이 드러나나

- 숨는 것: expert k+1의 확장은 expert k의 matmul(블록 여러 개면 그 전부) 아래에서 돈다. DMA는 한 expert 앞서 가므로 확장 시점엔 끝나 있다.
- 드러나는 것: 호출의 **첫 expert gate_up 확장**이다. 기다릴 앞 matmul이 없기 때문이다. int4의 DMA_FIRST와 같은 성격이다.
- 판정은 프로파일로 한다(`NNTR_HTP_PROFILE=2`).
  - `drain a+b`(DRAIN+DRAIN_DN): 기다린 시간이다. 확장이 덜 끝나 기다린 몫도 여기 들어간다. **이게 노출분이다.**
  - `int2 expand (worker)`: 워커가 확장에 쓴 시간의 합이다. 숨었든 아니든 **일의 양**이다. residual에서 빼지 않는다.
  - 숨겨졌다는 판정: int2의 drain이 int4의 drain보다 크지 않으면 숨은 것이다. expand가 수백 us여도 상관없다.
- 걱정되는 경우는 decode(M=1)다. expert당 matmul이 블록 1개라 짧다. 확장이 그보다 길면 drain이 커진다.
  - 손잡이: `MOE_EXPAND_SPLIT`(단위 크기)와 워커 수.
  - ponytail: staging을 1개로 줄이면 VTCM 1.42 MiB를 아끼지만 push가 확장 완료를 기다리게 된다. 지금은 2개다.

### 2.5 검증 (호스트, 실행함)

- `test/htp/host/run_host_checks.sh` 전부 통과. int2 값을 가진 같은 가중치를 WH와 WH2로 등록해 참조와 비교했다.
- background lane은 두 스케줄로 돌린다.
  - eager: submit 즉시 실행해서 가장 이른 스케줄이다. VTCM 영역을 너무 일찍 덮어쓰는 버그를 잡는다.
  - lazy: wait가 도달해야 실행해서 가장 늦은 스케줄이다. 새 스텁 모드 `g_bg_lazy`이고, 덜 기다림·엉뚱한 job·staging 조기 재사용을 잡는다.
- 변이 시험 다섯 개가 모두 검사를 실패시킨다: gate_up 덜 기다림, down 덜 기다림, 엉뚱한 job 기다림, 쓰는 중인 staging 재사용, 다음 down push 누락.
  - 처음엔 "엉뚱한 job"이 통과했다. 원인은 lazy 스텁의 버그였다. 이미 끝난 job을 기다리면 큐 전체를 돌려 버렸고, 고친 뒤 잡혔다.
- `tools/make_wh2_from_wh4.py --self-test`: 재패킹을 확장 규칙과 직접 colsum으로 확인한다. Gemma 파일 크기 재현 = **12,935,608,440 B**로, 55 §10의 12,336 MiB와 같다.
- `htp_compute_ops.cpp` 확인 범위:
  - `tools/htp_syntax_check.sh` 통과.
  - IDL에서 생성한 실제 시그니처로도 컴파일했다. 남은 오류는 호스트 remote.h 스텁에 없는 SDK 상수 2개뿐이다.
- **기기에서만 확인되는 것**: `Q6_W_vshuff_VVR` 바이트 순서(`ExpandWh2MatchesScalar`), 실제 동시성, 성능 전부.

## 3. 기기 가이드

PC 저장소는 `~/workspace/nntrainer`, 기기 앱 디렉터리는 `/data/local/tmp/nntrainer/causallm`이다. 기기가 여러 대면 먼저 `export ANDROID_SERIAL=<serial>`.

### 3.1 빌드 (IDL이 바뀌었으므로 stub → skel → 앱 → skel push 전부)

```bash
export HEXAGON_SDK_ROOT=$HOME/workspace/Hexagon_SDK/6.4.0.2 HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon ANDROID_NDK=$HOME/workspace/android-ndk-r26d
export DEFAULT_HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/$(ls $HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/ | sort -V | tail -1)
cd ~/workspace/nntrainer && git pull origin claude/epic-hopper-occf31 && git log --oneline -1
nntrainer/tensor/htp_backend/generate_stub.sh | tail -1 && grep -c "set_expert_bits\|expand_wh2" nntrainer/tensor/htp_backend/generated/nntr_hvx.h
(cd test/htp && ./build.sh 2>&1 | tail -1) && md5sum test/htp/build/libnntr_hvx_skel.so | cut -c1-8
cd Applications/CausalLM && ./build_android.sh --htp 2>&1 | tail -1 && ./install_android.sh 2>&1 | tail -1 && adb push ~/workspace/nntrainer/test/htp/build/libnntr_hvx_skel.so ~/workspace/nntrainer/test/htp/g4_int2_bench.sh /data/local/tmp/nntrainer/causallm/ && adb shell 'md5sum /data/local/tmp/nntrainer/causallm/libnntr_hvx_skel.so' | cut -c1-8
```
- 기대 결과:
  - `git log` 첫 줄이 이 문서 커밋이다.
  - grep 개수는 `2` 이상이다.
  - skel md5 앞 8자리가 PC와 기기에서 **같다**.
- 실패하면:
  - skel 빌드가 `DEFAULT_HEXAGON_TOOLS_ROOT`로 죽으면 두 번째 줄 export가 빠진 것이다.
  - `hvx_expand_wh2.c`에서 컴파일 오류가 나면 전체 출력을 보내 주세요. 이 파일은 기기 컴파일러에 처음 들어간다.

### 3.2 기기 gtest (먼저: 확장 바이트 순서와 int2 = int4 비트 일치)

```bash
cd ~/workspace/nntrainer && bash test/htp/run_u8i4_layer_on_device.sh 2>&1 | tee ~/g4_int2_gtest.log | grep -E "expand_wh2|moe_int2|ExpandWh2|Int2Matches|FAILED|PASSED" | head -20
```
- 기대 결과:
  - `path=expand_wh2 field=bad_bytes value=0 of 8192`
  - `path=moe_int2 field=bad_elems value=0 of 360448 nonzero=…`(0보다 큼)
  - 두 테스트 `[ OK ]`, 전체 PASSED.
- 실패하면:
  - `expand_wh2 bad_bytes`가 **8192 전부**(짝·홀 바이트가 뒤바뀜)면 `hvx_expand_wh2.c`의 `Q6_W_vshuff_VVR(odd, even, -1)` 두 인자를 바꾸면 된다. 그 숫자를 보내 주세요.
  - 일부만 틀리면 `first=` 값까지 보내 주세요.
  - `moe_int2`만 틀리면 파이프라인 대기 문제다. bad_elems 숫자를 보내 주세요.
  - **이 단계가 통과하기 전엔 §3.4 측정이 의미 없다.**

### 3.3 더미 파일 만들기 (PC, 약 10~20분, 디스크 약 24.1 GiB)

int4 파일에서 int2 값 두 벌을 만든다. `nntr_gemma4_wh2.bin`(int2 실행용)과 `nntr_gemma4_i4same.bin`(같은 값의 int4 레이아웃, int4 실행용)이다. 둘은 확장 후 같은 모델이라 라우팅·캐시 적중이 같고, **출력 텍스트도 같아야 한다**. 이게 기기 e2e 정확성 검사도 겸한다.

```bash
df -h ~/workspace | tail -1
adb pull /data/local/tmp/nntrainer/causallm/models/gemma4-26b-a4b-qs4cx-wh/config.json /tmp/g4_config.json
mkdir -p ~/workspace/g4dummy && cd ~/workspace/nntrainer && python3 tools/make_wh2_from_wh4.py --self-test && python3 tools/make_wh2_from_wh4.py --config /tmp/g4_config.json --in ~/workspace/nntrainer/Applications/CausalLM/res/gemma4_26ba4b/q40/nntr_gemma4_q40_arm.bin --out-wh2 ~/workspace/g4dummy/nntr_gemma4_wh2.bin --out-wh4 ~/workspace/g4dummy/nntr_gemma4_i4same.bin 2>&1 | tail -3; ls -l ~/workspace/g4dummy/
```
- 기대 결과:
  - self-test가 `12935608440 bytes`를 출력한다.
  - 본 실행이 `layout matches the file: 12935608440 bytes, 7680 expert weights`, 진행 줄, `done`을 출력한다.
  - 두 파일 모두 12935608440 B다.
- 실패하면:
  - `layout replay gives … the file has …`이면 아무것도 쓰지 않은 것이다. 양자화 때 `--fc_dtype`/`--embd_dtype`이 Q4_0이 아니었다면 `--fc-dtype`/`--embd-dtype`로 그 값을 주세요. 그래도 안 맞으면 두 숫자를 보내 주세요.
  - 디스크가 모자라면 `--out-wh4`를 빼고 int2 파일만 만듭니다. 그러면 int4 기준은 기존 실제 파일이 되고, 라우팅이 달라져 출력 비교는 못 합니다.

### 3.4 기기에 올리기 (모델 디렉터리 2개 + 파일 2개, 기기 여유 약 24.1 GiB 필요)

```bash
adb shell 'cd /data/local/tmp/nntrainer/causallm && sh g4_int2_bench.sh setup'
adb push ~/workspace/g4dummy/nntr_gemma4_wh2.bin /data/local/tmp/nntrainer/causallm/models/g4-i2/ && adb push ~/workspace/g4dummy/nntr_gemma4_i4same.bin /data/local/tmp/nntrainer/causallm/models/g4-i4same/
adb shell 'ls -l /data/local/tmp/nntrainer/causallm/models/g4-i2/*.bin /data/local/tmp/nntrainer/causallm/models/g4-i4same/*.bin; df -h /data | tail -1'
```
- 기대 결과:
  - setup이 `=== models/g4-i2: "model_file_name": "nntr_gemma4_wh2.bin" "num_to_generate": 128` 형식으로 두 줄을 출력한다.
  - 두 .bin이 12935608440 B다.
- 실패하면:
  - setup이 `cp` 오류를 내면 기존 모델 디렉터리 이름이 다른 것이다. `adb shell ls /data/local/tmp/nntrainer/causallm/models`를 보내 주세요.
  - 공간이 모자라면 int2 파일만 올리고 §3.5에서 `quick`으로 돌린 뒤 알려 주세요. i4 실행은 기존 디렉터리로 따로 돌립니다.

### 3.5 측정

```bash
adb shell 'cd /data/local/tmp/nntrainer/causallm && sh g4_int2_bench.sh' 2>&1 | tee ~/g4_int2_bench.log
```
- 대략 12회 실행이다(설정 5개 × cold/warm + 프로파일 2회). 회당 1~2분이다. 빨리 보려면 끝에 ` quick`을 붙이면 cold와 C=32를 건너뛴다.
- 기대 결과: 설정마다 다음이 나온다.
  - `=== <이름> pass=… pagecache=…`
  - `exit=0 text_md5=xxxxxxxx`
  - `prefill:`, `generation:`, `peak memory:` 줄
  - `[HTP] arena chunk … mapped total …`(마지막 청크)
  - `expert cache misses` / `expert prefetch` 줄(prefetch=off면 prefetch 줄은 없다)
  - 끝에 `=== same text int4/int2 (pf, nopf): A A B B` 형식이 나온다. **i4와 i2의 md5가 짝마다 같아야 한다.**
- 실패하면:
  - md5가 다르면 int2 경로가 다른 값을 낸 것이다. `adb pull /data/local/tmp/g4b ~/g4b`로 로그를 받아 보내 주세요.
  - `exit=`가 0이 아니면 그 설정의 `.err` 끝 30줄을 보내 주세요(`adb shell tail -30 /data/local/tmp/g4b/<이름>_warm.err`).
  - `pagecache=NOT_dropped`는 root가 아니라 page cache를 못 비웠다는 뜻이다. 그 cold는 진짜 cold가 아니다. 두 파일을 번갈아 쓰므로 일부만 남아 있다.
- 로그 파일 `~/g4_int2_bench.log` 전체를 보내 주시면 §4 표를 채웁니다.

## 4. 측정 매트릭스와 판정

| 이름 | 파일 | bits | C | read-ahead | 보는 것 |
|---|---|---|---|---|---|
| i4_pf | i4same | 4 | 16 | 켬(기본) | 기준 |
| i2_pf | wh2 | 2 | 16 | 켬 | flash 바이트 절반의 효과, arena 704 MiB |
| i4_nopf | i4same | 4 | 16 | 끔 (`NNTR_MOE_PREFETCH=0`) | read-ahead 없이 miss가 전부 동기 |
| i2_nopf | wh2 | 2 | 16 | 끔 | 같은 조건에서 int2 |
| i2_c32 | wh2 | 2 | 32 | 켬 | int4 C=16과 같은 arena로 캐시 2배 |
| i4_prof / i2_prof | | | 16 | 켬 | `NNTR_HTP_PROFILE=2` 단계 분해 |

- read-ahead는 prefill에서만 돈다(lfm2_moe_layer.cpp `expertPrefetchDepth`). decode의 miss는 어느 모드든 동기다. 그래서 두 모드 차이는 주로 prefill에 나타난다.
- 판정 1, 정확성: 짝마다 text md5가 같아야 한다.
- 판정 2, 숨김: 프로파일에서 `drain a+b`가 i2 ≤ i4면 확장이 숨은 것이다. `int2 expand (worker)`는 일의 양으로만 기록한다.
- 판정 3, e2e:
  - prefill ms와 decode TPS를 i4와 i2로 비교한다.
  - miss당 read ms가 약 절반이 되는지 본다(산술: 3.01 → 1.52 MB).
  - `weight DMA KB/call`이 절반인지 본다.
- 기대치(산술, 미측정): C=16 cold prefill 바닥 3.37 → 1.71 s. TPS와 확장 시간은 **추정하지 않는다**.

## 5. 측정 기록

### 5.1 기기 gtest (2026-10-06, 사용자 기기, `run_u8i4_layer_on_device.sh`)

- `ExpandWh2MatchesScalar`: `bad_bytes 0 of 8192`. `Q6_W_vshuff_VVR(odd, even, -1)` 바이트 순서가 맞다.
- `MoeLayerInt2MatchesInt4`: `bad_elems 0 of 360448`, nonzero 332222. WH2 경로(staging, 확장, 대기)가 gemma 형상·GeGLU에서 WH 경로와 비트 단위로 같다.
- 전체 PASSED (35 / 21 / 6 / 1). 성능은 아직 미측정이다.

### 5.2 e2e

| 이름 | pass | prefill ms | decode TPS | peak RSS KB | arena MiB | misses (ms/miss) | prefetch 수 (노출 ms) | text md5 |
|---|---|---|---|---|---|---|---|---|
| (기기 미측정) | | | | | | | | |

프로파일 (decode M==1 / prefill, us/call):

| 이름 | dsp | drain a+b | int2 expand (worker) | weight DMA KB/call | mm |
|---|---|---|---|---|---|
| (기기 미측정) | | | | | |
