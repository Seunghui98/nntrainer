#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

def debug_embedding():
    print("Loading model...")
    model = SentenceTransformer(
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        trust_remote_code=True,
        model_kwargs={
            "torch_dtype": torch.float32,
        },
    )
    model.max_seq_length = 512

    def hook_fn(module, input, output):
        name = module.__class__.__name__

        if not isinstance(output, dict):
            return

        # ---- (A) Pooling 입력 mask/토큰 보고 싶으면: Pooling 모듈에서 input dict 확인 ----
        if 'Pooling' in name:
            in0 = input[0] if (len(input) > 0 and isinstance(input[0], dict)) else None

            if in0 is not None and 'attention_mask' in in0:
                attn = in0['attention_mask']  # [B, L]
                print('!!!!')
                print(attn)
                print(f"\n[DEBUG Python] Pooling input attention_mask shape: {attn.shape}")
                print("  attention_mask[0][:50] =", attn[0, :50].detach().cpu().tolist())
                print("  valid_token_count =", int(attn[0].sum().item()))

        # ---- (B) token_embeddings가 있다면 masked mean을 재현해서 출력 ----
        if 'token_embeddings' in output and 'Pooling' in name:
            tokens = output['token_embeddings']  # [B, L, D]
            print(f"\n[DEBUG Python] Layer: {name} (Token Embeddings to Pooling)")
            print("  token_embeddings shape:", tuple(tokens.shape))

            # Pooling input dict에서 attention_mask 가져오기
            in0 = input[0] if (len(input) > 0 and isinstance(input[0], dict)) else None
            attn = in0.get('attention_mask', None) if in0 is not None else None

            # batch 0만 비교
            batch_tokens = tokens[0]  # [L, D]
            L, D = batch_tokens.shape

            # 1) 네가 하던 "그냥 평균" (mask 무시)
            sum_all = batch_tokens.sum(dim=0)
            pooled_all = sum_all / L
            print("  [NO MASK] pooled_all[:5] =", pooled_all[:5].detach().cpu().numpy())

            # 2) mask 기반 mean pooling 재현
            if attn is not None:
                mask = attn[0].to(batch_tokens.dtype).unsqueeze(-1)  # [L, 1]
                masked_tokens = batch_tokens * mask                  # [L, D]
                denom = mask.sum().clamp(min=1.0)                    # scalar
                pooled_masked = masked_tokens.sum(dim=0) / denom     # [D]

                print("  [MASKED] denom(valid_tokens) =", float(denom.item()))
                print("  [MASKED] pooled_masked[:5] =", pooled_masked[:5].detach().cpu().numpy())

                # 토큰 몇 개/마스크 값 같이 찍기 (앞 10개)
                for pos in range(min(L, 10)):
                    m = int(attn[0, pos].item())
                    v = batch_tokens[pos, :5].detach().cpu().numpy()
                    print(f"    pos={pos:02d} mask={m} token[:5]={v}")

        # ---- (C) Pooling 결과 sentence_embedding도 같이 찍기 ----
        if 'sentence_embedding' in output and ('Pooling' in name or 'Normalize' in name):
            emb = output['sentence_embedding']
            print(f"\n[DEBUG Python] Layer: {name} (sentence_embedding)")
            print(f"Shape: {emb.shape}")
            print("Values (first 10):", emb[0, :10].detach().cpu().numpy())
    
    print("\n[DEBUG] Model Structure and Hook Registration:")
    for idx, module in enumerate(model):
        name = module.__class__.__name__
        print(f"  {idx}: {name}")
        # Hook every module that might be related to pooling
        if any(x in name for x in ['Pooling', 'Normalize', 'Transformer']):
            module.register_forward_hook(hook_fn)
            print(f"    -> Hook registered")

    sentences = ["This is an example sentence", "Good Morning"]
    
    for sentence in sentences:
        print(f"\n[DEBUG] Tokenization for: '{sentence}'")
        ids = model.tokenizer.encode(sentence)
        tokens = model.tokenizer.tokenize(sentence)
        print(f"  Tokens: {tokens}")
        print(f"  IDs   : {ids}")
        print(f"  Count : {len(ids)}")

    print("\n--- Running Encode ---")
    # batch_size=1 to make comparison easier with CausalLM default
    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=True,
    )

    print("\n[DEBUG Python] Layer: Final (Embedding Normalize Output)")
    print(f"Shape: {embeddings.shape}")
    print("Values (first 10):", embeddings[0, :10])

if __name__ == "__main__":
    debug_embedding()
