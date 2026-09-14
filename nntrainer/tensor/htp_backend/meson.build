htp_backend_sources = [
  'htp_backend.cpp',
  'htp_compute_ops.cpp',
]

htp_backend_headers = [
  'htp_backend.h',
  'htp_rpcmem.h',
]

foreach s : htp_backend_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : htp_backend_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

# FastRPC client stub for the same IDL the DSP skel (test/htp/build.sh) is
# built from. Not a custom_target: qaic is a proprietary Hexagon SDK tool a
# normal build host does not have, and test/htp/build.sh already established
# the convention of a manual generation step for it. generate_stub.sh puts
# its output where this checks for it; both nntrainer_sources (this build)
# and jni/meson.build's ndk-build source list (the Android build) read
# nntrainer_sources by absolute path, so one entry here reaches both.
nntr_hvx_stub_c = meson.current_source_dir() / 'generated' / 'nntr_hvx_stub.c'
nntr_hvx_idl = meson.project_source_root() / 'test' / 'htp' / 'nntr_hvx.idl'
if not fs.is_file(nntr_hvx_stub_c)
  error('enable-htp=true requires the generated FastRPC stub. Run:\n' +
        '  HEXAGON_SDK_ROOT=<path> ' +
        meson.current_source_dir() / 'generate_stub.sh')
endif
# A MISSING stub is caught above; a STALE one used to sail through and fail
# later as an undeclared identifier in htp_compute_ops.cpp, which names the
# symbol but not the reason. Worse, test/htp/build.sh regenerates a SECOND
# copy under test/htp/generated for the DSP skel, so running only that one
# leaves a fresh skel against a stale client -- the device then rejects the
# call with EBADPARM and the build looks fine. Compare against the IDL both
# are generated from and say so here instead.
if fs.is_file(nntr_hvx_idl)
  stub_h = meson.current_source_dir() / 'generated' / 'nntr_hvx.h'
  if fs.is_file(stub_h)
    idl_newer = run_command(
      'test', nntr_hvx_idl, '-nt', stub_h, check: false).returncode() == 0
    if idl_newer
      error('The generated FastRPC stub is older than ' + nntr_hvx_idl +
            '.\nRegenerate the ARM client stub (test/htp/build.sh only does ' +
            'the DSP skel):\n' +
            '  HEXAGON_SDK_ROOT=<path> ' +
            meson.current_source_dir() / 'generate_stub.sh')
    endif
  endif
endif
nntrainer_sources += nntr_hvx_stub_c
nntrainer_inc_abs += meson.current_source_dir() / 'generated'
nntrainer_inc += include_directories('generated')
