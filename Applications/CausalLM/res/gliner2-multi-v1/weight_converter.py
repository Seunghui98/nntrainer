# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>

# @file weight_converter.py
# @brief weight conversion script for gliner2-multi-v1
# @author Seunghui Lee <shsh1004.lee@samsung.com>

import argparse
import torch
import numpy as np
from gliner2 import GLiNER2
from transformers import AutoConfig

def save_gliner2_multi_v1_for_nntrainer(params, n_layers, dtype, file):  
    """Convert and save weights as nntrainer format for multi-head attention model"""  
      
    def save_weight(weight):
        if torch.is_tensor(weight):
            weight = weight.detach().cpu().numpy()
        weight.astype(dtype).tofile(file)

    def save_projection(layer_name, proj_name):  
        """Helper function to handle base/lora weight saving"""  
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"  
        if lora_key in params:  
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))  
        else:  
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))

    def save_dense(name):
        """Helper to save standard weight and bias"""
        save_weight(params[f"{name}.weight"].permute(1, 0))
        save_weight(params[f"{name}.bias"])

    def save_layernorm(name):
        """Helper to save LayerNorm weight and bias"""
        save_weight(params[f"{name}.weight"])
        save_weight(params[f"{name}.bias"])

    def save_attention(layer_name, layer_idx):  
        # Save Q/K/V projections using helper  
        for proj in ["query_proj", "key_proj", "value_proj"]:
            save_projection(layer_name, f"attention.self.{proj}")
            save_weight(params[f"{layer_name}attention.self.{proj}.bias"])
        
        if layer_idx == 0:
            save_weight(params["encoder.encoder.rel_embeddings.weight"])
            # encoder.LayerNorm in PyTorch is actually the LayerNorm for relative embeddings
            save_layernorm("encoder.encoder.LayerNorm")
        
        save_dense(f"{layer_name}attention.output.dense")
        save_layernorm(f"{layer_name}attention.output.LayerNorm")

    # Save embedding layer  
    save_weight(params["encoder.embeddings.word_embeddings.weight"])
    save_layernorm("encoder.embeddings.LayerNorm")

    # Process all encoder layers  
    for layer_idx in range(n_layers):  
        layer_prefix = f"encoder.encoder.layer.{layer_idx}."  
        save_attention(layer_prefix, layer_idx)
        save_dense(f"{layer_prefix}intermediate.dense")
        save_dense(f"{layer_prefix}output.dense")
        save_layernorm(f"{layer_prefix}output.LayerNorm")
    
    for module in ["classifier"]:
        for i in [0, 2]:
            save_projection(f"{module}.{i}", "")
            save_weight(params[f"{module}.{i}.bias"])
    
    # Process classifier and count_pred
    for module in ["count_pred"]:
        for i in [0, 2]:
            save_projection(f"{module}.{i}", "")
            save_weight(params[f"{module}.{i}.bias"])
    
    save_weight(params["count_embed.pos_embedding.weight"])

    # Process span rep layers
    for module in ["project_end", "project_start", "out_project"]:
        for i in [0, 3]:
            save_projection(f"span_rep.span_rep_layer.{module}.{i}", "")
            save_weight(params[f"span_rep.span_rep_layer.{module}.{i}.bias"])
    
    # --- GRU Weight Processing ---
    # PyTorch(R, Z, N) -> NNTrainer(Z, R, N)
    
    # 1. Weight IH (Input-Hidden)
    w_ih = params["count_embed.gru.weight_ih_l0"]
    r, z, n = w_ih.chunk(3, dim=0)         # (Reset, Update, New) for pytorch weight order
    w_ih_new = torch.cat((z, r, n), dim=0) # -> (Z, R, N) for nntrianer gru
    save_weight(w_ih_new.permute(1, 0))  

    # 2. Weight HH (Hidden-Hidden)
    w_hh = params["count_embed.gru.weight_hh_l0"]
    r, z, n = w_hh.chunk(3, dim=0)         # (Reset, Update, New) for pytorch weight order
    w_hh_new = torch.cat((z, r, n), dim=0) # -> (Z, R, N) for nntrianer gru
    save_weight(w_hh_new.permute(1, 0)) 

    # 3. Bias IH
    b_ih = params["count_embed.gru.bias_ih_l0"]
    r, z, n = b_ih.chunk(3, dim=0)         # (Reset, Update, New) for pytorch weight order
    b_ih_new = torch.cat((z, r, n), dim=0) # -> (Z, R, N) for nntrianer gru
    save_weight(b_ih_new)

    # 4. Bias HH
    b_hh = params["count_embed.gru.bias_hh_l0"]
    r, z, n = b_hh.chunk(3, dim=0)         # (Reset, Update, New) for pytorch weight order
    b_hh_new = torch.cat((z, r, n), dim=0) # -> (Z, R, N) for nntrianer gru
    save_weight(b_hh_new) 
    
    for module in ["count_embed.projector"]:
        for i in [0, 2]:
            save_projection(f"{module}.{i}", "")
            save_weight(params[f"{module}.{i}.bias"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="fastino/gliner2-multi-v1")
    parser.add_argument("--output_name", type=str, default="./nntr_gliner2_multi_v1_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()
    
    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = GLiNER2.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path, subfolder="encoder_config")
    model = GLiNER2.from_pretrained(model_path, trust_remote_code=True, torch_dtype=data_dtype)
    model.eval()    

    print(model)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    
    with open(output_name, "wb") as f_model :
        save_gliner2_multi_v1_for_nntrainer(model.state_dict(), config.num_hidden_layers, data_dtype, f_model)
