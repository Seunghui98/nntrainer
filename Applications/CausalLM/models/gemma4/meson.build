gemma4_src = [
    meson.current_source_dir() / 'gemma4_causallm.cpp',
]

gemma4_inc = include_directories('.')

causallm_src += gemma4_src
causallm_inc += gemma4_inc
# the MoE block is the LFM2 MoE layer (router_type softmax_scale)
causallm_layer_dependencies += [lfm2_moe_layer_dep]
