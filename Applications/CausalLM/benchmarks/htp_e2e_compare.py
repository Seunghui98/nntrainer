#!/usr/bin/env python3
"""Compare CPU vs HTP greedy token generation for Qwen3-0.6B on device."""
import json, re, subprocess, sys, tempfile, os

DEVICE = "R3CY205ZMND"
MODEL_DIR = "/data/local/tmp/qwen3-0.6b"

def adb(*args):
    return subprocess.run(["adb", "-s", DEVICE, *args],
                          capture_output=True, text=True)

def set_engine(engine):
    """Pull nntr_config.json, set compute_engine, push back."""
    r = adb("shell", f"cat {MODEL_DIR}/nntr_config.json")
    cfg = json.loads(r.stdout)
    cfg["compute_engine"] = engine
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f); tmp = f.name
    adb("push", tmp, f"{MODEL_DIR}/nntr_config.json")
    os.unlink(tmp)

def run(engine):
    set_engine(engine)
    cmd = (f"cd /data/local/tmp && HTP_E2E_DUMP_IDS=1 LD_LIBRARY_PATH=/data/local/tmp "
           f"ADSP_LIBRARY_PATH=/data/local/tmp ./nntrainer_causallm {MODEL_DIR}")
    out = adb("shell", cmd).stdout
    m = re.search(r"TOKEN_IDS:\s*([\d,]+)", out)
    if not m:
        print(f"[{engine}] no TOKEN_IDS line:\n{out}"); sys.exit(2)
    return [int(x) for x in m.group(1).split(",") if x]

def main():
    cpu = run("cpu")
    htp = run("htp")
    print(f"CPU ({len(cpu)} ids): {cpu}")
    print(f"HTP ({len(htp)} ids): {htp}")
    if cpu == htp:
        print("PASS: token sequences identical"); return 0
    n = min(len(cpu), len(htp))
    div = next((i for i in range(n) if cpu[i] != htp[i]), n)
    print(f"FAIL: first divergence at token #{div}")
    print(f"  CPU={cpu[div] if div < len(cpu) else 'EOS'} "
          f"HTP={htp[div] if div < len(htp) else 'EOS'}")
    print("  (near-tie 여부는 디바이스 로그의 top-1/top-2 logit 격차로 판단 — README 참고)")
    return 1

if __name__ == "__main__":
    sys.exit(main())
