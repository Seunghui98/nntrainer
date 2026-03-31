// SPDX-License-Identifier: Apache-2.0
/**
 * @file   nntr_runner.cpp
 * @brief  Generic NNTrainer model runner for TorchFXConverter output.
 *
 * Loads any converter-produced INI via loadFromConfig() and runs one
 * inference pass, optionally with user-supplied token IDs.
 *
 * Usage:
 *   nntr_runner <output_dir/>
 *   nntr_runner <output_dir/> --input "1 2 3 4"
 *   nntr_runner <output_dir/> --input-file tokens.txt
 *   nntr_runner --ini m.ini [--json m.json] [--weights m.bin]
 *              [--input "1 2 3"] [--input-file tokens.txt]
 *
 * Exit codes:
 *   0  success
 *   1  error (details on stderr)
 */

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <app_context.h>
#include <engine.h>
#include <model.h>
#include <tensor_dim.h>

// CausalLM custom layers
#include <embedding_layer.h>
#include <mha_core.h>
#include <reshaped_rms_norm.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

using ModelHandle = std::unique_ptr<ml::train::Model>;

// ---------------------------------------------------------------------------
// String / file helpers
// ---------------------------------------------------------------------------

static bool endsWith(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string readFile(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static bool isDirectory(const std::string &path) {
  DIR *d = opendir(path.c_str());
  if (d) {
    closedir(d);
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Directory auto-discovery
// ---------------------------------------------------------------------------

struct FileSet {
  std::string ini;
  std::string json;
  std::string weights;
};

static FileSet discoverFiles(const std::string &dir) {
  FileSet fs;
  DIR *d = opendir(dir.c_str());
  if (!d)
    throw std::runtime_error("Cannot open directory: " + dir);

  struct dirent *ent;
  while ((ent = readdir(d)) != nullptr) {
    std::string name = ent->d_name;
    if (name == "." || name == "..")
      continue;
    std::string full = dir + "/" + name;

    if (fs.ini.empty() && endsWith(name, ".ini"))
      fs.ini = full;
    else if (fs.json.empty() && endsWith(name, ".json"))
      fs.json = full;
    else if (fs.weights.empty() &&
             (endsWith(name, ".bin") || endsWith(name, ".safetensors")))
      fs.weights = full;
  }
  closedir(d);
  return fs;
}

// ---------------------------------------------------------------------------
// Minimal JSON field extraction (no external dependency)
// ---------------------------------------------------------------------------

static std::string jsonExtract(const std::string &json,
                                const std::string &key) {
  std::string pattern = "\"" + key + "\"\\s*:\\s*([^,}\\]]+)";
  std::regex re(pattern);
  std::smatch m;
  if (std::regex_search(json, m, re)) {
    std::string val = m[1].str();
    val.erase(0, val.find_first_not_of(" \t\r\n\""));
    auto last = val.find_last_not_of(" \t\r\n\"");
    if (last != std::string::npos)
      val = val.substr(0, last + 1);
    return val;
  }
  return "";
}

static void printJsonMetadata(const std::string &json_path) {
  std::string c = readFile(json_path);
  if (c.empty())
    return;
  std::cout << "[nntr_runner] Model metadata:\n";
  auto p = [&](const char *label, const char *key) {
    std::string v = jsonExtract(c, key);
    if (!v.empty())
      std::cout << "  " << label << ": " << v << "\n";
  };
  p("model_type",        "model_type");
  p("arch_type",         "arch_type");
  p("hidden_size",       "hidden_size");
  p("num_layers",        "num_layers");
  p("num_heads",         "num_heads");
  p("num_kv_heads",      "num_kv_heads");
  p("head_dim",          "head_dim");
  p("intermediate_size", "intermediate_size");
  p("vocab_size",        "vocab_size");
  p("max_position_embeddings", "max_position_embeddings");
}

// ---------------------------------------------------------------------------
// Custom layer registration
// ---------------------------------------------------------------------------

static void registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto *app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::MHACoreLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::RMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::SwiGLULayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::TieWordEmbedding>);
  } catch (const std::invalid_argument &e) {
    std::cerr << "[nntr_runner] registerFactory warning: " << e.what() << "\n";
  }
}

// ---------------------------------------------------------------------------
// Token input helpers
// ---------------------------------------------------------------------------

/** Parse space/comma-separated integer token IDs from a string. */
static std::vector<int> parseTokens(const std::string &s) {
  std::vector<int> ids;
  std::istringstream iss(s);
  std::string tok;
  while (iss >> tok) {
    // strip trailing commas
    if (!tok.empty() && tok.back() == ',')
      tok.pop_back();
    if (!tok.empty())
      ids.push_back(std::stoi(tok));
  }
  return ids;
}

/** Read token IDs from a file (one per line, or space-separated). */
static std::vector<int> parseTokensFromFile(const std::string &path) {
  std::string content = readFile(path);
  if (content.empty())
    throw std::runtime_error("Cannot read token file: " + path);
  // replace newlines with spaces and parse
  std::replace(content.begin(), content.end(), '\n', ' ');
  return parseTokens(content);
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/**
 * Load model, compile, initialize, optionally load weights, run inference.
 *
 * @param ini_path   NNTrainer INI config.
 * @param bin_path   Weight binary (empty = skip loading weights).
 * @param arch_type  "encoder_only", "decoder_only", or "" (auto).
 * @param user_tokens User-supplied token IDs (empty = zero-filled dummy).
 */
static int runModel(const std::string &ini_path,
                    const std::string &bin_path,
                    const std::string &arch_type,
                    const std::vector<int> &user_tokens) {
  std::cout << "[nntr_runner] Loading config: " << ini_path << "\n";

  ModelHandle model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  if (model->loadFromConfig(ini_path) != 0) {
    std::cerr << "[nntr_runner] ERROR: loadFromConfig failed\n";
    return 1;
  }

  try {
    if (model->compile(ml::train::ExecutionMode::INFERENCE) != 0) {
      std::cerr << "[nntr_runner] ERROR: compile returned non-zero\n";
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "[nntr_runner] ERROR: compile threw: " << e.what() << "\n";
    return 1;
  }

  try {
    if (model->initialize(ml::train::ExecutionMode::INFERENCE) != 0) {
      std::cerr << "[nntr_runner] ERROR: initialize returned non-zero\n";
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "[nntr_runner] ERROR: initialize threw: " << e.what() << "\n";
    return 1;
  }

  if (!bin_path.empty()) {
    std::cout << "[nntr_runner] Loading weights: " << bin_path << "\n";
    try {
      model->load(bin_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    } catch (const std::exception &e) {
      std::cerr << "[nntr_runner] ERROR: weight load failed: " << e.what()
                << "\n";
      return 1;
    }
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  // ------------------------------------------------------------------
  // Build input buffer from getInputDimension()
  // ------------------------------------------------------------------
  std::vector<ml::train::TensorDim> input_dims;
  try {
    input_dims = model->getInputDimension();
  } catch (const std::exception &e) {
    std::cerr << "[nntr_runner] WARNING: getInputDimension failed: "
              << e.what() << " — skipping inference\n";
    std::cout << "[nntr_runner] Model loaded successfully (no inference).\n";
    return 0;
  }

  if (input_dims.empty()) {
    std::cerr << "[nntr_runner] WARNING: no input dims — skipping inference\n";
    std::cout << "[nntr_runner] Model loaded successfully (no inference).\n";
    return 0;
  }

  // Prepare float buffer with token IDs (or zeros for dummy run)
  std::vector<std::vector<float>> buffers;
  std::vector<float *> input_ptrs;
  for (size_t idx = 0; idx < input_dims.size(); ++idx) {
    size_t sz = input_dims[idx].getDataLen();
    buffers.emplace_back(sz, 0.0f);
    // fill with user-provided or sequential dummy tokens
    if (!user_tokens.empty()) {
      size_t copy_n = std::min(sz, user_tokens.size());
      for (size_t i = 0; i < copy_n; ++i)
        buffers.back()[i] = static_cast<float>(user_tokens[i]);
    } else {
      // dummy: sequential token IDs starting at 1
      for (size_t i = 0; i < sz; ++i)
        buffers.back()[i] = static_cast<float>(i + 1);
    }
    input_ptrs.push_back(buffers.back().data());
  }

  // ------------------------------------------------------------------
  // Run inference
  // ------------------------------------------------------------------
  bool is_decoder = (arch_type.find("decoder") != std::string::npos);
  std::cout << "[nntr_runner] Running "
            << (is_decoder ? "incremental" : "forward") << " inference ("
            << (user_tokens.empty() ? "dummy" : "user") << " input)...\n";

  try {
    if (is_decoder) {
      // Decoder-only: use incremental_inference (seq prefill pass)
      unsigned int seq_len =
        static_cast<unsigned int>(input_dims[0].getDataLen());
      std::vector<float *> labels;
      auto outputs = model->incremental_inference(
        1, input_ptrs, labels, seq_len, 0, seq_len, false);
      std::cout << "[nntr_runner] Inference OK — " << outputs.size()
                << " output tensor(s).\n";
    } else {
      // Encoder-only / pooling: standard inference
      auto outputs = model->inference(1, input_ptrs);
      std::cout << "[nntr_runner] Inference OK — " << outputs.size()
                << " output tensor(s).\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "[nntr_runner] ERROR: inference failed: " << e.what() << "\n";
    return 1;
  }

  std::cout << "[nntr_runner] Done.\n";
  return 0;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
  std::string ini;
  std::string json;
  std::string weights;
  std::string input_str;    // "--input 1 2 3"
  std::string input_file;   // "--input-file tokens.txt"
};

static void printUsage(const char *prog) {
  std::cerr
    << "Usage:\n"
    << "  " << prog << " <output_dir/> [options]\n"
    << "  " << prog << " --ini m.ini [--json m.json] [--weights m.bin] [options]\n"
    << "\nOptions:\n"
    << "  --input \"1 2 3 ...\"   Token IDs for inference input\n"
    << "  --input-file FILE       Read token IDs from file (one per line or"
       " space-separated)\n"
    << "  --ini FILE              NNTrainer INI config\n"
    << "  --json FILE             Converter JSON metadata (optional)\n"
    << "  --weights FILE          Weight binary .bin/.safetensors (optional)\n";
}

static Args parseArgs(int argc, char *argv[]) {
  Args args;
  std::map<std::string, std::string *> flags = {
    {"--ini",        &args.ini},
    {"--json",       &args.json},
    {"--weights",    &args.weights},
    {"--input",      &args.input_str},
    {"--input-file", &args.input_file},
  };

  std::vector<std::string> positional;
  bool any_flag = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto it = flags.find(a);
    if (it != flags.end()) {
      if (i + 1 >= argc)
        throw std::invalid_argument("Missing value after " + a);
      *it->second = argv[++i];
      any_flag = true;
    } else {
      positional.push_back(a);
    }
  }

  // Process positional args
  for (const auto &p : positional) {
    if (args.ini.empty() && isDirectory(p)) {
      FileSet fs = discoverFiles(p);
      args.ini     = fs.ini;
      args.json    = fs.json;
      args.weights = fs.weights;
    } else if (args.ini.empty() && endsWith(p, ".ini")) {
      args.ini = p;
    } else if (args.weights.empty() &&
               (endsWith(p, ".bin") || endsWith(p, ".safetensors"))) {
      args.weights = p;
    } else if (args.json.empty() && endsWith(p, ".json")) {
      args.json = p;
    }
  }
  (void)any_flag;
  return args;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  Args args;
  try {
    args = parseArgs(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "[nntr_runner] Argument error: " << e.what() << "\n";
    printUsage(argv[0]);
    return 1;
  }

  if (args.ini.empty()) {
    std::cerr << "[nntr_runner] ERROR: No INI file found.\n";
    printUsage(argv[0]);
    return 1;
  }

  // Determine arch_type from JSON metadata
  std::string arch_type;
  if (!args.json.empty()) {
    std::string content = readFile(args.json);
    if (!content.empty()) {
      arch_type = jsonExtract(content, "arch_type");
      printJsonMetadata(args.json);
    }
  }

  // Parse user tokens
  std::vector<int> user_tokens;
  try {
    if (!args.input_file.empty()) {
      user_tokens = parseTokensFromFile(args.input_file);
      std::cout << "[nntr_runner] Input: " << user_tokens.size()
                << " tokens from file\n";
    } else if (!args.input_str.empty()) {
      user_tokens = parseTokens(args.input_str);
      std::cout << "[nntr_runner] Input: " << user_tokens.size()
                << " tokens from --input\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "[nntr_runner] ERROR parsing tokens: " << e.what() << "\n";
    return 1;
  }

  try {
    registerCustomLayers();
    return runModel(args.ini, args.weights, arch_type, user_tokens);
  } catch (const std::exception &e) {
    std::cerr << "[nntr_runner] FATAL: " << e.what() << "\n";
    return 1;
  }
}
