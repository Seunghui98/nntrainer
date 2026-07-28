# Standalone, self-contained example: no nntrainer symbols are used on the host
# build (the sdkl NPU path is compiled only under ENABLE_HEXKL on Android via
# the flat Android.mk). So the host target needs no nntrainer dependency.
e = executable('hexkl_fc_compare',
  ['hexkl_fc_compare.cpp'],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)
