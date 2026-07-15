#!/usr/bin/env python3
"""Compare CPU vs HTP greedy token generation for Qwen3-0.6B on device."""
import json, re, subprocess, sys, tempfile, os

DEVICE = os.environ.get("HTP_E2E_DEVICE", "R3CY205ZMND")
INSTALL_DIR = os.environ.get("HTP_E2E_INSTALL_DIR",
                             "/data/local/tmp/nntrainer/causallm")
MODEL_DIR = os.environ.get("HTP_E2E_MODEL_DIR", f"{INSTALL_DIR}/models/qwen3-0.6b")

def adb(*args):
    return subprocess.run(["adb", "-s", DEVICE, *args],
                          capture_output=True, text=True)

def _config_path():
    """main.cpp prefers nntr_config_quantized.json over nntr_config.json when
    both exist (see Applications/CausalLM/main.cpp); match that here so we
    edit the config the binary actually loads."""
    quantized = f"{MODEL_DIR}/nntr_config_quantized.json"
    probe = adb("shell", "test", "-f", quantized)
    if probe.returncode == 0:
        return quantized
    return f"{MODEL_DIR}/nntr_config.json"

def set_engine(engine):
    """Pull the active nntr_config*.json, set compute_engine, push back."""
    cfg_path = _config_path()
    r = adb("shell", f"cat {cfg_path}")
    cfg = json.loads(r.stdout)
    cfg["compute_engine"] = engine
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f); tmp = f.name
    adb("push", tmp, cfg_path)
    os.unlink(tmp)

def build_run_cmd(model_dir):
    return (f"cd {INSTALL_DIR} && HTP_E2E_DUMP_IDS=1 "
            f"LD_LIBRARY_PATH={INSTALL_DIR} "
            f"ADSP_LIBRARY_PATH={INSTALL_DIR} "
            f"./nntrainer_causallm {model_dir}")

def parse_token_ids(stdout):
    m = re.search(r"TOKEN_IDS:\s*([\d,]+)", stdout)
    if not m:
        return []
    return [int(x) for x in m.group(1).split(",") if x]

def parse_generated_text(stdout):
    m = re.search(r"GENERATED_TEXT_BEGIN\n(.*?)\nGENERATED_TEXT_END",
                  stdout, re.DOTALL)
    return m.group(1).strip() if m else ""

def run(engine):
    set_engine(engine)
    out = adb("shell", build_run_cmd(MODEL_DIR)).stdout
    ids = parse_token_ids(out)
    if not ids:
        print(f"[{engine}] no TOKEN_IDS line:\n{out}"); sys.exit(2)
    return ids, parse_generated_text(out)

def main():
    cpu_ids, cpu_txt = run("cpu")
    htp_ids, htp_txt = run("htp")

    print("=== CPU (golden) generated text ===")
    print(cpu_txt)
    print("=== HTP decode-on generated text ===")
    print(htp_txt)

    print(f"\nCPU  ({len(cpu_ids)} ids): {cpu_ids}")
    print(f"HTP  ({len(htp_ids)} ids): {htp_ids}")
    if cpu_ids == htp_ids:
        print("token ids identical (reference)")
    else:
        n = min(len(cpu_ids), len(htp_ids))
        div = next((i for i in range(n) if cpu_ids[i] != htp_ids[i]), n)
        print(f"first token-id divergence at #{div} (reference)")
    print("\nHuman decision: read the two sentences above and judge whether "
          "the HTP output is equivalent to the CPU golden.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
