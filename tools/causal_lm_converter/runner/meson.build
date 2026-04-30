# tools/causal_lm_converter/runner/meson.build
#
# Builds the generic INI + safetensors runner. Compiles the causal-LM custom
# layer sources directly so this tool stays independent of the heavier
# Applications/CausalLM build (which pulls in tokenizer libraries).

# Path to the causal-LM layer source/include tree, expressed relative to this
# meson.build file. We compile the layer .cpp files directly into the runner
# binary so this tool stays independent of the larger Applications/CausalLM
# build (which links against a tokenizer library).
clmc_layers_relpath = '../../../Applications/CausalLM/layers'

# Reusable importer library.
# Other apps (tokenizer-driven generators, JNI bridges, benchmarks) can link
# against this static lib + its include dir to import converter artifacts
# without copy-pasting the runner code.
clmc_importer_sources = [
  'causal_lm_importer.cpp',
  clmc_layers_relpath + '/rms_norm.cpp',
  clmc_layers_relpath + '/reshaped_rms_norm.cpp',
  clmc_layers_relpath + '/swiglu.cpp',
  clmc_layers_relpath + '/mha_core.cpp',
  clmc_layers_relpath + '/embedding_layer.cpp',
  clmc_layers_relpath + '/tie_word_embedding.cpp',
]

clmc_layers_inc = include_directories(clmc_layers_relpath)
clmc_importer_inc = include_directories('.')

causal_lm_importer_lib = static_library(
  'causal_lm_importer',
  clmc_importer_sources,
  include_directories: [clmc_importer_inc, clmc_layers_inc],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
  install: false,
)

causal_lm_importer_dep = declare_dependency(
  link_with: causal_lm_importer_lib,
  include_directories: [clmc_importer_inc],
)

# CLI runner: thin wrapper around the importer.
executable('causal_lm_runner',
  ['runner.cpp'],
  dependencies: [
    causal_lm_importer_dep, nntrainer_dep, nntrainer_ccapi_dep, openmp_dep,
  ],
  install: false,
)

# Minimal embedded example (documentation-as-code). Builds together with the
# runner so a broken importer header surfaces immediately.
executable('causal_lm_example_embedded',
  ['example_embedded.cpp'],
  dependencies: [
    causal_lm_importer_dep, nntrainer_dep, nntrainer_ccapi_dep, openmp_dep,
  ],
  install: false,
)
