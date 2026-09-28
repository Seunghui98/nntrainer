# Golden logits of the LFM2.5-shaped LFM2-MoE fixture through the in-process HTP build

Produced by `test/htp/host/run_inproc_e2e.sh` with `NNTR_INPROC_GOLDEN=update`,
like `../lfm2_moe_tiny`: the fixture
`test/unittest/models/causallm_reference/lfm2_moe_tiny_lfm25` (LFM2.5-8B-A1B's
per-layer shape at six layers: hidden 2048, 32 q heads, 8 kv heads, head_dim
64, max_position_embeddings 2048, rope_theta 5e6, layers
conv/conv/attention/conv/attention/conv with two dense layers, MoE
intermediate 256, 4 experts top-2; the same generator at `--dim 2048
--n-heads 32 --n-kv-heads 8 --head-dim 64 --max-pos 2048 --layer-types
conv,conv,attention,conv,attention,conv --num-dense 2 --rope-theta 5000000
--moe-inter 256`, plan 136 section 2), stream-quantized with `--fc_dtype Q4_0
--moe_dtype QS4CX_WH` on the host ISA, run with **prompt 512**, 8 greedy
tokens, `--max-seq 2048`, `moe_engine: htp`, the per-token switch **off**.

Logits only (`logits_<step>.f32`, step 0 = prefill), no `manifest.txt` and no
MoE call dumps: at prompt 512 one prefill call's input is 4 MiB, and the two
small fixtures already pin every call's bytes. `run_inproc_e2e.sh` requires a
byte-identical match (`E2E eval golden-lfm25 ... bit_identical=1`); what a
mismatch means is the tiny golden's README. The fixture exists because the
device fault of #136 shows only at the real shape and prompt length (a KV seed
of 512 rows, RoPE past 512, the conv state seed after a long CPU prefill):
the harness compares the switch-on run against the switch-off run of this
fixture under the SNR floor (stopping at the first router top-k flip, a
near-tie on random weights) and the token policy, and this golden pins the
switch-off run itself.
