// SPDX-License-Identifier: Apache-2.0
/**
 * @file   runner.cpp
 * @brief  Thin CLI wrapper around ``causal_lm::Importer`` for build-and-run
 *         validation of converted causal-LM models.
 *
 * The interesting logic lives in ``causal_lm_importer.{h,cpp}`` and is
 * intended to be reused by other applications (a tokenizer-driven generator,
 * an Android JNI bridge, a benchmark harness, etc.). This binary exists so
 * that:
 *   - Tests have something concrete to invoke.
 *   - End users have a one-shot "does my converted model load and forward?"
 *     diagnostic without writing any C++ themselves.
 *   - Quick interactive checks: feed a tokenized prompt and see the top-K
 *     next-token candidates, or auto-generate N tokens greedily.
 *
 * Pipeline (delegated to causal_lm::import_model):
 *   register_custom_layers -> createModel -> setProperty -> loadFromConfig
 *   -> compile -> initialize -> load(safetensors)
 *
 * Input handling:
 *   - Default: feed all-zero token IDs (vocab-bounds-safe for any model)
 *   - --input-tokens "1,2,3,..."         : explicit token IDs (comma-separated)
 *   - --input-tokens-file <path>         : one integer per line / whitespace
 *
 * Output handling:
 *   - Always prints the first 8 logits of position 0 (smoke check).
 *   - --top-k N   : print top-N tokens (id + logit) for the LAST input
 *                   position -- this is what generation actually consumes.
 *   - --generate N: greedy-decode N tokens after the prompt and print them
 *                   as ``generated_tokens: 12,34,...``. Generation is done
 *                   inside the runner (not Python) so the model's KV cache
 *                   is preserved across steps.
 *   - --print-shape : print full output buffer length (debug aid).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "causal_lm_importer.h"

namespace {

void usage(const char *argv0) {
  std::cerr
    << "Usage: " << argv0
    << " <model.ini> <model.safetensors> <nntr_config.json> [options]\n"
    << "Options:\n"
    << "  --no-forward             skip the forward pass (load-only smoke)\n"
    << "  --input-tokens IDS       comma-separated token IDs to feed in\n"
    << "                           (length must match init_seq_len; padded\n"
    << "                            with 0s and warned if shorter)\n"
    << "  --input-tokens-file PATH read whitespace-separated token IDs from a file\n"
    << "  --prompt-len N           number of *real* prompt tokens (the rest are\n"
    << "                           padding). Required for --generate so the\n"
    << "                           loop knows where to write generated tokens.\n"
    << "                           Defaults to ``init_seq_len`` when omitted.\n"
    << "  --top-k N                print top-N next-token candidates for the\n"
    << "                           LAST input position (default: 0 = off)\n"
    << "  --generate N             after the prompt, greedy-decode N tokens.\n"
    << "                           Prints ``generated_tokens: 12,34,...``\n"
    << "  --print-shape            print the raw output buffer length\n";
}

/** Parse comma-separated token IDs from a string into a flat vector<int>. */
std::vector<int> parse_token_list(const std::string &s) {
  std::vector<int> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() &&
           (s[i] == ',' || s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) {
      ++i;
    }
    if (i >= s.size()) break;
    size_t j = i;
    while (j < s.size() && s[j] != ',' && s[j] != ' ' && s[j] != '\t' &&
           s[j] != '\n') {
      ++j;
    }
    out.push_back(std::stoi(s.substr(i, j - i)));
    i = j;
  }
  return out;
}

/** Read whitespace-separated token IDs from a file. */
std::vector<int> read_token_file(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<int> out;
  int v;
  while (f >> v) out.push_back(v);
  return out;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    usage(argv[0]);
    return 1;
  }
  causal_lm::ImportOptions opts;
  opts.ini_path = argv[1];
  opts.weights_path = argv[2];
  opts.runtime_config_path = argv[3];
  opts.execution_mode = ml::train::ExecutionMode::INFERENCE;

  bool run_forward = true;
  int top_k = 0;
  int generate_n = 0;
  int prompt_len = -1;  // -1 = "use init_seq_len"
  bool print_shape = false;
  std::vector<int> token_input;

  for (int i = 4; i < argc; ++i) {
    auto a = std::string(argv[i]);
    if (a == "--no-forward") {
      run_forward = false;
    } else if (a == "--top-k" && i + 1 < argc) {
      top_k = std::stoi(argv[++i]);
    } else if (a == "--generate" && i + 1 < argc) {
      generate_n = std::stoi(argv[++i]);
    } else if (a == "--prompt-len" && i + 1 < argc) {
      prompt_len = std::stoi(argv[++i]);
    } else if (a == "--print-shape") {
      print_shape = true;
    } else if (a == "--input-tokens" && i + 1 < argc) {
      token_input = parse_token_list(argv[++i]);
    } else if (a == "--input-tokens-file" && i + 1 < argc) {
      token_input = read_token_file(argv[++i]);
    } else {
      std::cerr << "Unknown option: " << a << std::endl;
      usage(argv[0]);
      return 1;
    }
  }

  try {
    causal_lm::register_custom_layers();
    auto imported = causal_lm::import_model(opts);
    auto &model = *imported.model;
    const auto &rc = imported.runtime;

    std::cout << "[runner] ini=" << opts.ini_path << "\n"
              << "[runner] weights=" << opts.weights_path << "\n"
              << "[runner] batch=" << rc.batch_size
              << " init_seq=" << rc.init_seq_len
              << " max_seq=" << rc.max_seq_len
              << " tensor_type=" << rc.model_tensor_type << std::endl;
    std::cout << "[runner] weights loaded" << std::endl;

    if (!run_forward) {
      std::cout << "[runner] OK (no forward)" << std::endl;
      return 0;
    }

    // Build the input float buffer. The graph compiled in INI uses
    // ``input_shape = 1:1:init_seq_len``, so the buffer is
    // ``batch * init_seq_len`` floats. Token IDs are stored as floats and
    // cast back to size_t inside the embedding layer.
    const size_t need = static_cast<size_t>(rc.batch_size) * rc.init_seq_len;
    std::vector<float> input(need, 0.0f);
    if (!token_input.empty()) {
      if (token_input.size() > need) {
        std::cerr << "[runner] WARNING: " << token_input.size()
                  << " tokens provided but only " << need
                  << " fit; truncating." << std::endl;
      } else if (token_input.size() < need) {
        std::cerr << "[runner] note: only " << token_input.size()
                  << " tokens provided (" << need
                  << " expected); padding with 0." << std::endl;
      }
      const size_t n = std::min(token_input.size(), need);
      for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(token_input[i]);
      }
      std::cout << "[runner] fed " << n << " tokens";
      // Print the first few IDs so users can sanity-check the input.
      const size_t preview = std::min<size_t>(n, 16);
      std::cout << " (first " << preview << ":";
      for (size_t i = 0; i < preview; ++i) std::cout << " " << token_input[i];
      std::cout << ")" << std::endl;
    } else {
      std::cout << "[runner] no --input-tokens; feeding all-zeros." << std::endl;
    }
    std::vector<float *> inputs = {input.data()};

    auto outputs = model.incremental_inference(
      rc.batch_size, inputs, /*labels=*/{}, rc.init_seq_len, 0,
      rc.init_seq_len);

    if (outputs.empty() || outputs.front() == nullptr) {
      throw std::runtime_error("forward returned no outputs");
    }
    float *out = outputs.front();

    // We do not know the vocab from the runner alone, so we read it back
    // from the lm_head section in the runtime: outputs is laid out as
    // ``[batch, seq_len, vocab]``. ``incremental_inference(0..init_seq)``
    // returns the full seq_len's worth, so vocab = total / (batch*seq).
    // The total length is not directly exposed; we approximate by parsing
    // the lm_head section via the model API. For simplicity we infer vocab
    // only when top-K is requested and let the smoke check stay shape-free.
    //
    // Heuristic: walk back from the model's last layer's output dim.
    // Use the API: model->getOutputDimension() if available; otherwise we
    // fall back to inspecting the runtime config's tokenizer hints.
    //
    // For the prevailing case (Qwen3 with tie_word_embeddings), vocab is
    // huge (151936), so we cannot enumerate logits cheaply -- top-K is the
    // right primitive.

    // Smoke check: first 8 logits should be finite.
    constexpr int kSample = 8;
    bool all_finite = true;
    for (int i = 0; i < kSample; ++i) {
      if (!std::isfinite(out[i])) {
        all_finite = false;
        break;
      }
    }
    std::cout << "[runner] forward OK, first " << kSample
              << " logits (position 0):";
    for (int i = 0; i < kSample; ++i) std::cout << " " << out[i];
    std::cout << std::endl;
    if (!all_finite) {
      std::cerr << "[runner] FAIL: non-finite values in output" << std::endl;
      return 2;
    }

    if (top_k > 0 || print_shape) {
      // Use vocab from the runtime config (the Python converter records it).
      // We deliberately do not call model->getOutputDimension(), because
      // INI-loaded graphs lack an explicit output node and that call throws.
      if (rc.vocab_size == 0) {
        throw std::runtime_error(
          "vocab_size missing from nntr_config.json; re-run the converter "
          "or pass --top-k 0 to skip vocab-aware decoding");
      }
      const size_t vocab = rc.vocab_size;

      // ``incremental_inference(B, in, lab, init_len, 0, init_len)`` returns
      // logits for the **last** token position only — i.e. one [batch, vocab]
      // row per call. This is what generation actually consumes (argmax /
      // sample over the last-position logits to pick the next token).
      if (print_shape) {
        std::cout << "[runner] output shape: batch=" << rc.batch_size
                  << " vocab=" << vocab
                  << " (total floats per call="
                  << static_cast<size_t>(rc.batch_size) * vocab << ")"
                  << std::endl;
      }

      if (top_k > 0) {
        // For batch=1 the layout is simply [vocab]; for larger batches the
        // caller would pick which row to inspect.
        const size_t row_offset = 0;
        std::vector<std::pair<float, int>> scored(vocab);
        for (size_t v = 0; v < vocab; ++v) {
          scored[v] = {out[row_offset + v], static_cast<int>(v)};
        }
        const int n = std::min<int>(top_k, static_cast<int>(vocab));
        std::partial_sort(scored.begin(), scored.begin() + n, scored.end(),
                          [](auto &a, auto &b) { return a.first > b.first; });
        std::cout << "[runner] top-" << n
                  << " next-token logits (the row generation uses):"
                  << std::endl;
        for (int i = 0; i < n; ++i) {
          std::cout << "  rank " << (i + 1)
                    << ": token_id=" << scored[i].second
                    << "  logit=" << scored[i].first << std::endl;
        }
      }
    }

    // ---- Greedy generation loop ------------------------------------------
    // We must generate inside the runner (not from Python via subprocess
    // round-trips) so that the model's KV cache is preserved across steps.
    //
    // Step protocol matches Applications/CausalLM/models/causal_lm.cpp:
    //   1. The initial forward above processed positions [0, prompt_len).
    //   2. For each new token t in [prompt_len, prompt_len + N):
    //        - argmax(last_logits) -> next_token
    //        - write next_token into input[t] (the same buffer the graph
    //          reads from; the embedding layer reads input[t] for position t)
    //        - call incremental_inference(B, [input], {}, init_seq_len, t,
    //          t+1). The model uses the cached K/V from previous steps.
    //        - the returned output[0] is the logit row for position t.
    if (generate_n > 0) {
      if (rc.vocab_size == 0) {
        throw std::runtime_error(
          "vocab_size missing from nntr_config.json; "
          "--generate requires it for argmax");
      }
      if (prompt_len < 0) {
        // Default: assume the entire input is the prompt.
        prompt_len = static_cast<int>(rc.init_seq_len);
      }
      if (prompt_len <= 0 ||
          static_cast<size_t>(prompt_len) > rc.init_seq_len) {
        throw std::runtime_error(
          "--prompt-len out of range; must be in (0, init_seq_len]");
      }
      // The graph's input_shape is fixed at init_seq_len; the rest of the
      // buffer past prompt_len is where generated tokens get written. We
      // therefore can only generate up to (init_seq_len - prompt_len)
      // tokens before we'd run out of positions.
      const int max_gen =
        static_cast<int>(rc.init_seq_len) - prompt_len;
      if (generate_n > max_gen) {
        std::cerr << "[runner] WARNING: --generate " << generate_n
                  << " exceeds the " << max_gen
                  << " positions available after prompt_len="
                  << prompt_len << "; truncating." << std::endl;
        generate_n = max_gen;
      }

      std::vector<int> generated;
      generated.reserve(generate_n);

      // ``out`` already holds the last-position logit row for the prompt.
      // First generated token comes from picking argmax on it.
      auto argmax_vocab = [&](float *row) {
        int best = 0;
        float best_v = row[0];
        for (size_t v = 1; v < rc.vocab_size; ++v) {
          if (row[v] > best_v) { best_v = row[v]; best = static_cast<int>(v); }
        }
        return best;
      };

      int next_id = argmax_vocab(out);
      generated.push_back(next_id);

      // Generation step loop.
      // Position semantics: after the initial call we have logits "as if"
      // we just consumed position prompt_len-1. We now want logits at
      // position prompt_len (the slot the just-picked token occupies).
      for (int step = 0; step < generate_n - 1; ++step) {
        const int t = prompt_len + step;  // the token slot we are filling
        if (static_cast<size_t>(t) >= rc.init_seq_len) break;
        input[t] = static_cast<float>(next_id);

        std::vector<float *> step_inputs = {input.data()};
        auto step_out = model.incremental_inference(
          rc.batch_size, step_inputs, /*labels=*/{}, rc.init_seq_len,
          /*from=*/static_cast<unsigned int>(t),
          /*to=*/static_cast<unsigned int>(t + 1));
        if (step_out.empty() || step_out.front() == nullptr) {
          throw std::runtime_error("generation step returned no outputs");
        }
        next_id = argmax_vocab(step_out.front());
        generated.push_back(next_id);
      }

      // Print as a single CSV line so the Python wrapper can parse it
      // with one regex.
      std::cout << "generated_tokens:";
      for (size_t i = 0; i < generated.size(); ++i) {
        std::cout << (i ? "," : " ") << generated[i];
      }
      std::cout << std::endl;
    }

    std::cout << "[runner] OK" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[runner] ERROR: " << e.what() << std::endl;
    return 1;
  }
}
