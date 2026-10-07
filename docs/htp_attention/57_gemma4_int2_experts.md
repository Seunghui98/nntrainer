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

- **파일은 packed다.** `NNTR_MOE_EXPERT_BITS=2`에서 expert 가중치는 파일에 WH2 바이트(니블의 절반) + N f32 scale + N f32 colsum으로만 들어 있고, 뒤의 모든 텐서는 그만큼 앞당겨진다. Gemma 파일은 12,935,608,440 → **7,226,112,120 B (6.73 GiB)**.
- 로더(`neuralnet.cpp`의 .bin 오프셋 순회)는 같은 스위치 아래에서 **virtual QS4CX_WH** 텐서(= 스트리밍 expert)만 그 크기로 센다. 다른 텐서는 그대로다. 호스트 expert pool은 같은 바이트 수를 pread하고 scale·colsum을 바로 뒤에서 읽는다(`expertWhBytes`, htp_wh_layout.h).
- 스위치는 파일에 기록되지 않는다. int4 파일을 BITS=2로 열거나 그 반대면 첫 expert 뒤의 모든 오프셋이 틀어진다. ponytail: 파일 헤더가 없어 로더가 확인할 수 없다. 더미 도구가 쓴 크기(위 숫자)와 파일 크기가 같은지 눈으로 확인한다.
- 처음 구현(a77ccbe~570005d)은 int4 레이아웃을 유지하고 앞 절반만 읽는 형식이었다. 그 파일(13 GB)은 **이제 읽히지 않는다**. 저장 공간·push 시간·readahead 낭비 때문에 packed로 바꿨다(§5.2 첫 실행은 그 옛 형식으로 쟀다).
- 첫 expert를 놓기 전에 DSP에 `set_expert_bits(2)`를 한 번 보낸다. 이후 swap은 arena에서 WH2로 읽고, scale·colsum은 WH2 바로 뒤에서 읽는다.

| | int4 (WH) | int2 (WH2) |
|---|---|---|
| expert 칸 (gate_up+down, 4 KB 정렬) | 3,010,560 B (2.871 MiB) | 1,523,712 B (1.453 MiB) |
| miss 1회 flash 읽기 | 3.01 MB | 1.52 MB |
| expert 1개 DMA (DDR→VTCM) | 2.84 MiB | 1.42 MiB |
| C=16 arena | 1408 MiB | 704 MiB |
| C=32 arena | — | 1408 MiB (int4 C=16과 같음) |
| C=16 cold prefill 바닥 (55 §4 방식, 3.0 GB/s) | 10.12 GB → 3.37 s | 5.12 GB → 1.71 s |
| 모델 파일 | 12,935,608,440 B | 7,226,112,120 B |

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
  - 본 실행이 `layout matches the file: 12935608440 bytes, 7680 expert weights; packed int2 file will be 7226112120 bytes`, 진행 줄, `done: … = 7226112120 bytes`를 출력한다.
  - int2 파일은 7,226,112,120 B, int4(i4same) 파일은 12,935,608,440 B다.
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

**read-ahead 끔 / reader 수 (2026-10-06, packed int2, C=16, PROFILE=2)**

| 실행 | prefill | read-ahead | miss (전체) | reader ms/expert | 합산 읽기 속도 |
|---|---|---|---|---|---|
| 기본 (reader 4) | 3920 ms | 3360 읽음, 노출 21 ms | 328 | 3.61 | 1.68 GB/s |
| `NNTR_MOE_PREFETCH=0` | 4789 ms (+869) | 없음 | 935, 1.59 ms/miss, file read 1490 ms | — | 동기 0.96 GB/s |
| reader 2 | 4821 ms | 노출 390 ms | | 2.44 | 1.25 GB/s |
| reader 6 | 4817 ms | 노출 0 ms | | 3.53 | 2.58 GB/s |

판정:
- **읽기가 드러나면 비용은 실재한다**: read-ahead를 끄자 prefill +0.87 s, miss당 1.59 ms(1.52 MB). int4였다면 miss당 바이트가 2배다(미측정).
- **합산 읽기 속도는 reader 수에 비례해 오른다**(1.25 → 1.68 → 2.58 GB/s). flash 한계(3.0 GB/s)가 아니라 **reader 한 스레드의 한계**(0.4~0.6 GB/s, uncached arena로의 pread)다. reader 6이면 5.1 GB를 2.0 s에 읽어 여유가 생긴다.
- **실행 간 편차가 0.9 s다.** reader 2(노출 390 ms)와 reader 6(노출 0)이 둘 다 4.8 s로 같고, 기본 4개가 3.9 s였다. 읽기 노출로 설명되지 않는다 → 발열 또는 page cache 상태. 연속 실행의 뒤쪽일수록 느린 패턴과 맞다. **int4/int2 비교는 같은 조건을 번갈아 2~3회 반복해야 한다.** 미확인.
- 더미 라우팅은 prefill에서 층당 약 20 expert만 더 건드린다(미스 ~600). read-ahead는 다음 층 expert 전부(3360)를 읽으므로 더미에서는 5배 과잉 읽기다. 실제 모델은 prefill에서 거의 128개를 다 건드리므로 과잉이 아니다.

**packed int2, NNTR_HTP_PROFILE=2 (2026-10-06)** — C=16, read-ahead 기본, 512 tok prompt / 512 생성. 파일 7,226,112,120 B, arena 704 MiB, 출력 텍스트 정상(더미지만 프롬프트를 되풀이).

| | prefill (M>1, 31 calls) | decode (M==1, 15360 calls) |
|---|---|---|
| e2e | 3920 ms (130.6 TPS) | 46292 ms (11.06 TPS, 90.4 ms/tok) |
| MoE 호출 host / dsp | 21259 / 20585 us/call, 합 659 ms | 1529 / 1439 us/call, 합 23.5 s = decode의 51% |
| mm / acc / requant / quant | 9753 / 4493 / 1569 / 504 | 829 / 392 / 2 / 71 |
| drain (gu + dn) = 노출 대기 | 1601 + 947 = 2548 (12%) | 107 + 11 = 118 (8%) |
| int2 expand (worker, 일의 양) | 7888 | 1472 |
| weight DMA | 61968 KB/call | 11616 KB/call (8 × 1.42 MiB) |
| miss | 328회 전체, 1.44 ms/miss, file read 합 472 ms | |
| read-ahead | 3360 experts, 노출 21 ms | |

판정:
- **int2가 줄이는 것은 flash 읽기(miss)인데 이 실행에는 miss가 사실상 없다.** 328회, 전부 0.47 s. 더미 라우팅이 고정되어 거의 전부 캐시 적중이고, prefill 읽기는 read-ahead가 99% 숨긴다. 그래서 e2e가 int4와 같은 것이 맞다. 실제 라우팅(top-8/128, C=16)이면 decode에서 토큰당 수백 miss가 생기고 그때 1.44 ms/miss의 절반 효과가 나타난다. 미측정.
- **확장 노출 상한**: drain이 MoE 호출의 8%(decode)/12%(prefill), e2e로는 각각 ≤4%/≤2%. int4의 drain과 비교해야 "숨었다"를 말할 수 있다(i4same 미실행).
- **e2e 병목은 MoE 밖**: decode 90 ms 중 MoE HTP 46, 나머지 44 ms는 CPU(Q4_0 dense MLP·projection·lm_head 262144×2816). prefill 3.9 s 중 MoE 0.66 s + 등록 0.79 s, 나머지 2.5 s가 CPU. 54의 "projection/FFN을 HTP로"가 Gemma에는 아직 없다. int2가 비운 arena 704 MiB가 그 자리다.
- MoE 호출 안: decode acc_read 27%, mm 58%. M=1에서 HMX 64행 타일의 1행만 쓴다 — 구조적 비용.

**첫 int2 실행 (2026-10-06, 사용자 기기)** — `NNTR_MOE_EXPERT_BITS=2`, C=16, read-ahead 기본, `NNTR_HTP_PROFILE=1`, 8 threads, 447토큰 prompt, 512토큰 생성, 원본 int4 파일과의 같은 조건 비교는 아직 없음.

| 항목 | 값 |
|---|---|
| arena | 704 MiB (3청크) — §2.2 산술과 같음, int2 stride가 적용됨 |
| prefill | 447 tok, 3840 ms (116 TPS) |
| decode | 512 tok, 43739 ms (11.7 TPS), 85.4 ms/token |
| peak RSS | 3,306,556 KB |
| 등록 (첫 forward에 포함) | 480 weights, 790 ms (convert 1.57 ms/weight) |
| MoE prefill 콜 | 31 calls, 18.9 ms/call, 585 ms |
| MoE decode 콜 | 15360 calls, 1461.8 us/call, 22.45 s = decode의 51% (토큰당 30 × 1.46 = 43.9 ms) |
| expert miss | 331 (decode 콜당 0.02), 1.36 ms/miss 읽기 |
| read-ahead | 3360 experts, 노출 대기 21 ms, 99% 제때 도착, reader 3.61 ms/expert |

해석:
- 동작은 의도대로다. arena 704 MiB, set_expert_bits 성공, 오류 없음. 출력은 더미 가중치라 두 토큰 반복(의미 없음).
- **miss가 비정상적으로 적다.** 출력이 두 토큰 반복으로 무너져 라우팅이 거의 고정되었고, decode가 사실상 전부 캐시 적중이다. 실제 모델의 decode miss 조건을 대표하지 못한다. int4와의 비교는 같은 값의 int4 파일(i4same)로 해야 공정하다.
- decode 85 ms/token 중 MoE HTP 콜 43.9 ms, 나머지 약 41 ms는 MoE 밖(CPU attention, Q4_0 dense MLP, 262144 vocab lm_head, norm)이다.
- 확장이 숨었는지는 level 1로는 알 수 없다. `NNTR_HTP_PROFILE=2`의 drain과 `int2 expand (worker)`가 필요하다.
- miss당 1.36 ms(1.52 MB, 약 1.1 GB/s)는 산술 기대(int4 대비 절반)를 확인할 기준이 아직 없다. 같은 조건 int4 실행과 비교해야 한다.

| 이름 | pass | prefill ms | decode TPS | peak RSS KB | arena MiB | misses (ms/miss) | prefetch 수 (노출 ms) | text md5 |
|---|---|---|---|---|---|---|---|---|
| (기기 미측정) | | | | | | | | |

프로파일 (decode M==1 / prefill, us/call):

| 이름 | dsp | drain a+b | int2 expand (worker) | weight DMA KB/call | mm |
|---|---|---|---|---|---|
| (기기 미측정) | | | | | |

## 6. expert 적재 경로 최적화 (계획, 미측정)

§5.2에서 읽기 한계는 flash가 아니라 reader 스레드 한 개(0.4~0.6 GB/s)이고, 실행 간 편차가 0.9 s다. 그래서 순서는 (1) 편차 통제 → (2) 있는 손잡이 sweep → (3) 코드 변경이다.

### 6.1 측정 규칙
- 같은 설정을 2회 이상, 설정을 번갈아, 사이에 60 s 휴식. `test/htp/g4_expert_sweep.sh`가 이 규칙으로 돌고 온도도 찍는다.
- 판정 열: prefill ms, read-ahead 노출 ms, reader ms/expert(→ 합산 GB/s = reader 수 × 1.52 MB / ms), miss ms/miss.

### 6.2 손잡이 (코드 변경 없음)
| 손잡이 | 지금 | 볼 것 |
|---|---|---|
| `NNTR_MOE_PREFETCH_READERS` | 4 | 2/4/6/8: 합산 속도가 어디서 멈추나(flash 3.0 GB/s 또는 uncached 쓰기 4.9 GB/s). 계산 스레드와의 경합은 prefill ms로 |
| `NNTR_NUM_THREADS` | 8 | 6 + reader 6: 코어를 나눠 주면 전체가 빨라지나 |
| `NNTR_MOE_PREFETCH` (깊이) | 전부 | 1: 한 층 앞만. 메모리 압박·과잉 읽기 대비 노출 |
| `NNTR_MOE_CACHE_EXPERTS` | 16 | 32 (int2라 arena는 int4 C=16과 같음): 읽을 양 10% 감소, miss 감소 |
| `NNTR_MOE_PREFETCH_CPUS` | caller 제외 전부 | big 코어만 / little 코어만 |

### 6.3 코드 변경 후보 (sweep 결과를 보고 고른다)
1. **page cache 선읽기 분리**: 다음 층 expert에 `posix_fadvise(WILLNEED)`(또는 `readahead()`)를 걸어 flash→page cache는 커널이 깊은 큐로, page cache→arena 복사만 reader가 한다. reader 한 스레드의 QD1 한계를 우회한다. 가장 싸다.
2. **bounce buffer**: cached heap에 pread 후 arena로 큰 단위 memcpy. uncached 쓰기가 per-thread 한계라면 효과, flash라면 없음.
3. **slot 단위 한 번에 읽기**: gate_up+down이 파일에서 연속(expert당 1.52 MB 1회 pread)인지 확인. 지금은 2회.
4. **읽기 순서**: 라우팅이 많이 쓰는 expert부터(prefill은 전부 쓰므로 해당 없음), decode는 top-k 확률 순.
5. **read-ahead의 과잉 읽기**: 더미에서만 문제(§5.2). 실제 모델에선 해당 없음.

### 6.4 int4 비교
같은 sweep을 i4same 파일(`NNTR_MOE_EXPERT_BITS=4`)로 한 번 더 돌리면 모든 열이 int4/int2 쌍으로 나온다. 특히 reader ms/expert(바이트 2배)와 노출 ms.

## 7. 왜 TPS가 안 올랐나: expert 경로의 숨은 몫과 드러난 몫

int2 실측(§5.2)과 int4 산술로 expert 경로를 세 단계로 나눠 적는다. int4 열은 **같은 조건 미측정**이며 `test/htp/g4_io_dma_probe.sh`가 그 열을 채운다(int4 실제 모델 vs int2 더미, read-ahead 켬/끔 각 2회, 첫 회는 워밍업).

### 7.1 flash → arena (호스트 reader)

| | int2 실측 | int4 산술 |
|---|---|---|
| prefill에 읽는 양 | 3360 × 1.52 MB = 5.1 GB | 10.2 GB |
| reader 4개 합산 속도 | 1.68 GB/s (expert당 3.61 ms) | 같다고 보면 |
| reader가 읽는 데 쓴 시간 | **3.0 s** | **6.1 s** |
| prefill 전체 | 4.0 s | — |
| 드러난 대기 (read-ahead) | 21 ms | 6.1 > 4.0이면 **약 2 s 드러남** (page cache에 있지 않다면) |
| read-ahead 끔: 동기 miss | 1.59 ms/miss, prefill +0.87 s | miss당 2배 바이트 |

- int2에서 읽기는 **거의 전부 숨었다**(3.0 s 중 0.02 s 노출). 그래서 int2 prefill은 CPU 계산 시간(3.2 s) + MoE 호출(0.66 s)로 결정되고, 읽기 바이트를 절반으로 줄여도 시간이 안 변한다.
- int4가 "133 TPS"였다면 10 GB를 4 s 안에 읽었다는 뜻이므로 page cache(두 번째 실행 이후)에서 읽었을 가능성이 크다. cold면 int4는 읽기가 드러난다. **probe의 i4_pf 노출 ms가 이걸 가른다.**
- 한계는 flash가 아니라 reader 스레드 한 개(0.4~0.6 GB/s). reader 6개면 2.6 GB/s(§5.2).

### 7.2 arena → VTCM (DMA)

| | int2 실측 | int4 산술 |
|---|---|---|
| prefill 호출당 | 61,968 KB | 123,936 KB |
| HMX 동작 중 DMA 속도 (doc 51 §2.26) | ~12 GB/s | ~12 GB/s |
| prefill 호출당 DMA 시간 | 5.2 ms / 20.6 ms 호출 | 10.3 ms / ~20.6 ms |
| decode 호출당 | 11,616 KB → **1.0 ms** / 1.44 ms 호출 | 23,232 KB → **1.9 ms** / ~1.4 ms 호출 |
| drain (가중치 기다린 시간) | prefill 2.5 ms, decode 0.12 ms | **probe가 채움** |

- prefill: int4도 DMA가 호출 시간 안에 들어가므로(10.3 < 20.6) 절반이 되어도 시간 이득은 없다.
- decode: int4는 DMA 1.9 ms가 호출 1.4 ms보다 길어 **DMA bound 가능성**이 있다. int2는 1.0 ms로 들어가고 실측 drain 0.12 ms다. int4의 decode drain이 ~0.5 ms로 나오면 int2가 decode 호출당 0.4 ms(30층 → 토큰당 12 ms, 90 → 78 ms, 11.1 → 12.8 TPS)를 번 것이다. **이것이 int2가 지금 조건에서 시간으로 벌 수 있는 유일한 몫이고, probe의 i4 decode drain이 판정한다.**

### 7.3 VTCM 안 확장 (int2만)

| | prefill 호출 | decode 호출 |
|---|---|---|
| 워커 시간 (일의 양) | 7,888 us | 1,472 us |
| 드러난 상한 (drain 전체) | 2,548 us (12%) | 118 us (8%) |
| int4 drain과의 차이 | probe | probe |

- 확장이 추가한 노출은 많아야 drain 전체이고, 그중 int4에도 있던 몫(호출당 첫 expert 대기)을 빼야 순수 확장 노출이다.

### 7.4 결론 (현재 더미, C=16, 기기 warm)
- prefill TPS가 같은 이유: 읽기도 DMA도 int4부터 이미 숨어 있었고(또는 page cache), int2는 숨은 것을 더 숨겼을 뿐이다. prefill 시간은 CPU 3.2 s가 정한다.
- decode TPS가 같은 이유: miss가 없어(더미 라우팅) 읽기 이득이 0이고, DMA 이득은 int4 drain을 봐야 안다(최대 +1.7 TPS 산술).
- int2가 시간으로 보이는 조건: (a) cold 또는 page cache 밖 (b) 실제 라우팅의 decode miss (c) CPU 계산을 HTP로 옮겨 prefill이 읽기 바닥에 닿을 때.

### 7.5 probe 실행
```
cd /data/local/tmp/nntrainer/causallm && sh g4_io_dma_probe.sh models/gemma4-26b-a4b-qs4cx-wh models/gemma4-26b-a4b-int2-wh
```
요약 줄의 열: prefill_ms, decode_tps, read(prefetch_n, exposed_ms, reader_ms, miss, ms/miss), prefill call(dsp, drain, mm, dma_kb, expand), decode call(같음). rep=2끼리 비교한다.

## 8. prefill CPU 몫의 분해 (2026-10-07, `--profile` 빌드, 생성 1토큰, int2 C=16)

레이어 종류별 누적 (prefill 5.82 s + 1토큰 생성 0.45 s; 프로파일 빌드라 평소 4.0 s보다 느리고 비중만 본다):

| 종류 | ms | 비중 | 내용 |
|---|---:|---:|---|
| mha_core | 2249 | 36% | attention 본체 (QKᵀ, softmax, PV), CPU |
| fully_connected | 2110 | 34% | 층당 wq ≈20, attention_out ≈21, wk ≈6.5, (wv), ffn_gate/up/down ≈20 → **projection ≈1.4 s, dense MLP ≈0.6 s** (29·28층 표본으로 나눈 추정) |
| lfm2_moe | 1371 | 22% | MoE 층 전체(HTP 콜 0.66 s + 라우팅·read-ahead 등록·호스트 몫) |
| scalar_multiply + rms_norm + 기타 | 530 | 8% | |

판정:
- **attention 본체(mha_core)가 단일 최대다.** 54 §R7은 "attention은 CPU 유지"였는데, Gemma는 full 층 head_dim 512·sliding 256, 512토큰에서 층당 75 ms가 CPU에 남는다. HTP attention 커널(`hexkl_attn_u8`, 30~37 문서, 디바이스 검증: prefill에서 CPU 대비 43~65×)은 있으나 **`mha_core.cpp`에 연결되어 있지 않다**(ComputeOps 훅 없음).
- projection을 **FC 개별 콜**로 HTP에 보내는 것은 LFM2에서 **prefill 손해**로 닫혔다(50 §3.7: 콜당 포장 고정비). Gemma도 같다. 유효한 길은 50 §6의 "왕복 자체를 없애기" = **attention 블록 한 콜**(q/k/v projection + RoPE + attention + out projection)이다. 45 Phase C/D.
- dense MLP는 **블록 융합 콜**(51 §1: `dense_ffn_engine`, MoE 커널로 up·gate·down 한 콜, 10.4 ms/층 vs ARM 19.5)로 이득이 확인됐다. Gemma는 GeGLU라 `dense_ffn_layer`에 act 선택(이미 커널엔 `act` 인자가 있다)과 `Gemma4CausalLM`이 `dense_ffn_engine` 키를 읽게 하는 일이 남는다. 예상: 층당 −9 ms × 30 = **−0.3 s**.
- 순서: (1) dense_ffn GeGLU + Gemma 연결 — 작고 검증된 길, −0.3 s. (2) attention 블록 한 콜 — 큰 일(−2 s 이상 가능), 30/45 문서 설계 재사용. (3) 그 뒤 prefill은 읽기 바닥(reader 4개 3.0 s, 6개 2.0 s)에 닿고, 그때 int2와 reader 수가 시간으로 나타난다.

