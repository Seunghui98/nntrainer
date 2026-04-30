# Tools subdirectory.
#
# Each tool builds independently. The causal_lm_converter runner depends on
# the CausalLM custom layer sources, so we gate it on enable-transformer
# (the same flag that gates Applications/CausalLM).

if (get_option('platform') != 'tizen') and (get_option('platform') != 'windows')
  if get_option('enable-transformer')
    subdir('causal_lm_converter')
  endif
endif
