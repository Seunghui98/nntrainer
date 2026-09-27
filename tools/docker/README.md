# nntrainer Hexagon dev container

One `linux/amd64` image that builds nntrainer on x86 and cross-builds the
HTP backend (host checks, DSP skel, Android app). On `htp_moe` it is the
Mac client's stand-in for the Ubuntu workstation (contract
`docs/plans/0001-htp-moe-decode-agent-system.md` §4.1a); the phone is
driven by a human. Typical use:

```
tools/docker/run.sh bash -c 'source tools/htp/env.sh && bash test/htp/host/run_host_checks.sh'
tools/docker/run.sh bash -c 'source tools/htp/env.sh && ./test/htp/build.sh'   # needs HexKL beta.2 mounted
```


```
tools/docker/setup_wizard.sh          # one-time: runtime, image, SDK login, model files, smoke test
tools/docker/run.sh <command...>      # run anything inside the container, repo mounted at /work
tools/docker/run.sh                   # interactive shell
tools/docker/run.sh --build-image     # rebuild after editing the Dockerfile
```

Host layout the wrapper expects (all overridable by environment, see `run.sh`):

| Host path | In container | Contents |
|---|---|---|
| repo | `/work` | this checkout (build dirs are created inside it) |
| `~/Qualcomm/Hexagon_SDK/<ver>/` | `/opt/qcom/Hexagon_SDK/<ver>/` | Hexagon SDK installed by `qpm-cli` inside the container (licensed, never in the image) |
| `~/Qualcomm/hexkl_addon/` | `/opt/qcom/hexkl_addon/` | HexKL micro API (optional, HMX work) |
| `~/Qualcomm/hexkl_addon/` | `/opt/qcom/hexkl_addon/` | HexKL package root: `lib/<sdk ver>/hexagon_toolv19_v79/` or the flat `lib/hexagon_toolv19_v79/`; `htp_moe` needs **1.0 beta.2** |
| `Applications/CausalLM/res/lfm2_moe/lfm2-8b-a1b/` | `/model` (ro) | LFM2.5-8B-A1B model files when present (`MODEL_DIR` overrides) |

The entrypoint sources `setup_sdk_env.source` of the newest SDK version (or
`HEXAGON_SDK_VERSION`) so `HEXAGON_SDK_ROOT`, `DEFAULT_HEXAGON_TOOLS_ROOT`
and `DEFAULT_TOOLS_VARIANT` are set (the compiler itself is not on `PATH`;
the scripts call `$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang` by
full path and `run_sim_test.sh` picks the `run_main_on_hexagon` image of
`$DEFAULT_TOOLS_VARIANT`, e.g. `hexagon_toolv19_v75` for HEXAGON_Tools
19.0.04); the scripts under `tools/hexagon/` then work unchanged:

```
tools/docker/run.sh ./tools/hexagon/build_host_x86.sh
HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/build_sim_test.sh
HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/run_sim_test.sh profile acc
HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/build_skel.sh
tools/docker/run.sh ./tools/hexagon/build_host_test.sh
```

Notes

* The simulator itself is an x86 binary running under Rosetta/QEMU on Apple
  silicon; expect `profile acc` to take several times longer than the "few
  minutes" recorded on a native workstation.
* `build_skel.sh` defaults to `HEX_ARCH=v79`; the shipping skel is v75
  (HEXAGON.md §7), so always pass `HEX_ARCH` explicitly.
* clang-format-14 lives in the image: `tools/docker/run.sh clang-format-14 -i <files>`.
