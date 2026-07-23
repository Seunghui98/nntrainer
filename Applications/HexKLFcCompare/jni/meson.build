fc_compare_sources = ['hexkl_fc_compare.cpp']

fc_compare_deps = [nntrainer_dep, nntrainer_ccapi_dep]

e = executable('hexkl_fc_compare',
  fc_compare_sources,
  include_directories: include_directories('.'),
  dependencies: fc_compare_deps,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
