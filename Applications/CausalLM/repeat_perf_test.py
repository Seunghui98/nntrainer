import subprocess
import re
import time
import statistics
import sys

def get_thermal_temp():
    try:
        # thermal_zone0 is usually CPU or reliable ambient on many devices
        cmd = ["adb", "shell", "cat", "/sys/class/thermal/thermal_zone0/temp"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip()) / 1000.0
    except Exception as e:
        print(f"Error reading temp: {e}")
    return 0.0

def get_process_count():
    try:
        # Count lines containing nntrainer_causallm, excluding grep itself
        cmd = ["adb", "shell", "ps -ef | grep nntrainer_causallm | grep -v grep | wc -l"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception as e:
        print(f"Error counting processes: {e}")
    return 0

def set_cpu_governor(governor):
    print(f"Setting CPU governor to: {governor}")
    try:
        # Try to set for all CPUs (cpu0 to cpu7 typically)
        cmd = f"echo {governor} > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" 
        # Note: policy path varies, sometimes it is cpu*/cpufreq/scaling_governor
        # Let's try the wildcard approach via shell expansion
        cmd = f"for path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo {governor} > $path; done"
        
        subprocess.run(["adb", "shell", cmd], check=True)
        
        # Verify
        res = subprocess.run(["adb", "shell", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"], capture_output=True, text=True)
        print(f"Verification (cpu0): {res.stdout.strip()}")
        
    except Exception as e:
        print(f"Failed to set governor: {e}")

def run_test(run_id, model_path, omp_threads=None, taskset_mask=None):
    start_temp = get_thermal_temp()
    start_procs = get_process_count()
    print(f"[{run_id}] Starting run... (Temp: {start_temp:.1f}C, Procs: {start_procs})")
    
    export_cmd = f"export OMP_NUM_THREADS={omp_threads} && " if omp_threads else ""
    taskset_cmd = f"taskset {taskset_mask} " if taskset_mask else ""
    
    cmd = [
        "adb", "shell", 
        f"cd /data/local/tmp/nntrainer/causallm && {export_cmd}{taskset_cmd}./run_causallm.sh {model_path}"
    ]
    
    # Capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    end_temp = get_thermal_temp()
    end_procs = get_process_count()
    output = result.stdout
    
    # regex for TPS
    # Example: prefill: 437 tokens, 1203 ms, 363.259 TPS
    # Example: generation: 310 tokens, 5585 ms, 55.5058 TPS
    
    prefill_match = re.search(r"prefill:.*,\s+([\d\.]+)\s+TPS", output)
    gen_match = re.search(r"generation:.*,\s+([\d\.]+)\s+TPS", output)
    e2e_match = re.search(r"\[e2e time\]:\s+(\d+)\s+ms", output)
    
    prefill_tps = float(prefill_match.group(1)) if prefill_match else 0.0
    gen_tps = float(gen_match.group(1)) if gen_match else 0.0
    e2e_time = float(e2e_match.group(1)) if e2e_match else 0.0
    
    print(f"[{run_id}] Completed. Temp: {end_temp:.1f}C | Procs: {end_procs} | Prefill: {prefill_tps:.2f} | Gen: {gen_tps:.2f} | e2e: {e2e_time:.0f}ms")
    
    return {
        "prefill_tps": prefill_tps,
        "gen_tps": gen_tps,
        "e2e_time": e2e_time,
        "start_temp": start_temp,
        "end_temp": end_temp,
        "start_procs": start_procs,
        "end_procs": end_procs,
        "error": result.stderr if result.returncode != 0 else ""
    }

def main():
    model_path = "./models/qwen3-0.6b"
    omp_threads = None
    taskset_mask = None
    governor = None
    
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        model_path = args.pop(0)
        
    # Check for arguments
    for arg in args:
        if arg.startswith("--omp="):
            omp_threads = int(arg.split("=")[1])
        elif arg.startswith("--taskset="):
            taskset_mask = arg.split("=")[1]
        elif arg.startswith("--governor="):
            governor = arg.split("=")[1]
            
    num_runs = 10
    results = []
    
    print(f"Starting {num_runs} iterations for model: {model_path}")
    if omp_threads:
        print(f"Configuration: OMP_NUM_THREADS={omp_threads}")
    if taskset_mask:
        print(f"Configuration: taskset mask={taskset_mask}")
    if governor:
        set_cpu_governor(governor)
    print("-" * 50)
    
    try:
        for i in range(num_runs):
            res = run_test(i+1, model_path, omp_threads, taskset_mask)
            results.append(res)
            # Minimal sleep to separate process runs but not let device cool completely effectively
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    if not results:
        print("No results collected.")
        return

    # Filter valid runs
    prefills = [r["prefill_tps"] for r in results if r["prefill_tps"] > 0]
    gens = [r["gen_tps"] for r in results if r["gen_tps"] > 0]
    e2es = [r["e2e_time"] for r in results if r["e2e_time"] > 0]
    
    print("-" * 50)
    print("SUMMARY")
    print("-" * 50)
    
    if prefills:
        print(f"Runs captured: {len(prefills)}")
        print(f"Prefill TPS: Mean={statistics.mean(prefills):.2f}, StdDev={statistics.stdev(prefills) if len(prefills)>1 else 0:.2f}")
        print(f"             Min={min(prefills):.2f}, Max={max(prefills):.2f}")
    else:
        print("No valid prefill data found.")
        
    if gens:
        print(f"Gen TPS:     Mean={statistics.mean(gens):.2f}, StdDev={statistics.stdev(gens) if len(gens)>1 else 0:.2f}")
        print(f"             Min={min(gens):.2f}, Max={max(gens):.2f}")

    if e2es:
        print(f"E2E Time:    Mean={statistics.mean(e2es):.0f}ms, StdDev={statistics.stdev(e2es) if len(e2es)>1 else 0:.0f}ms")
        print(f"             Min={min(e2es):.0f}ms, Max={max(e2es):.0f}ms")
        
    print("\nDetailed Trend:")
    print("Run\tStartT\tEndT\tGenTPS\tPrefill\tE2E(ms)")
    for i, r in enumerate(results):
        print(f"{i+1}\t{r['start_temp']:.1f}\t{r['end_temp']:.1f}\t{r['gen_tps']:.2f}\t{r['prefill_tps']:.2f}\t{r['e2e_time']:.0f}")
                
if __name__ == "__main__":
    """
    How to use:
      python3 Applications/CausalLM/repeat_perf_test.py {model path to test} {options}
      python3 Applications/CausalLM/repeat_perf_test.py ./models/qwen3-0.6b --omp=4
    """
    main()

