# TorchFXRunner — generic NNTrainer model runner for TorchFXConverter output.
#
# Builds the nntr_runner executable that loads any model produced by
# converter.py via loadFromConfig(), without per-model C++ compilation.
#
# Usage:
#   nntr_runner <output_dir/>                          # auto-discover files
#   nntr_runner --ini m.ini [--json m.json] ...        # explicit flags
#   nntr_runner model.ini [model.bin]                  # positional (compat)

runner_inc = [
  include_directories('.'),
]

# Custom layer dependencies (same set as TorchFXConverter tests)
runner_layer_deps = [
  causallm_rms_norm_dep,
  causallm_swiglu_dep,
  causallm_mha_core_dep,
  causallm_embedding_layer_dep,
  causallm_reshaped_rms_norm_dep,
  causallm_tie_word_embedding_dep,
]

executable('nntr_runner',
  [meson.current_source_dir() / 'nntr_runner.cpp'],
  include_directories: runner_inc,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep, runner_layer_deps],
)
