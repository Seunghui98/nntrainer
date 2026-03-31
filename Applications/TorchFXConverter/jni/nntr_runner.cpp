// SPDX-License-Identifier: Apache-2.0
/**
 * @file   nntr_runner.cpp
 * @date   30 Mar 2026
 * @brief  Standalone NNTrainer model runner for TorchFXConverter output.
 *
 * Loads a NNTrainer model from an INI configuration file, registers all
 * CausalLM custom layers, compiles, initializes, and summarizes the model.
 * No per-model recompilation is needed — any converted INI can be run directly.
 *
 * The runner accepts the full set of files produced by converter.py:
 *   --ini      model.ini       NNTrainer INI config (required for loading)
 *   --json     model.json      Converter JSON metadata (optional, printed)
 *   --header   model.h         Auto-generated C++ header (optional, info only)
 *   --weights  model.bin       Pre-converted weight binary (optional)
 *
 * Positional-argument form (backward compatible):
 *   nntr_runner <model.ini>
 *   nntr_runner <model.ini> <weights.bin>
 *
 * Auto-detect form — pass any files, extensions are used to determine roles:
 *   nntr_runner model.ini model.json model.h model.bin
 *
 * Exit codes:
 *   0  — model loaded, compiled, and initialized successfully
 *   1  — error (printed to stderr)
 */

#include <algorithm>
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

// CausalLM custom layers
#include <embedding_layer.h>
#include <mha_core.h>
#include <reshaped_rms_norm.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

using ModelHandle = std::unique_ptr<ml::train::Model>;

// ---------------------------------------------------------------------------
// Helper: string utilities
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

// ---------------------------------------------------------------------------
// JSON metadata printer (minimal, no external dependency)
// ---------------------------------------------------------------------------

/**
 * @brief Extract a scalar value from a flat JSON string by key.
 *
 * Only handles top-level string/number/bool values in the "model" object.
 * A full JSON parser is not needed — converter output is well-formed.
 */
static std::string jsonExtract(const std::string &json,
                                const std::string &key) {
  // Look for "key": <value>
  std::string pattern = "\"" + key + "\"\\s*:\\s*([^,}\\]]+)";
  std::regex re(pattern);
  std::smatch m;
  if (std::regex_search(json, m, re)) {
    std::string val = m[1].str();
    // Strip surrounding quotes and whitespace
    val.erase(0, val.find_first_not_of(" \t\r\n\""));
    auto last = val.find_last_not_of(" \t\r\n\"");
    if (last != std::string::npos)
      val = val.substr(0, last + 1);
    return val;
  }
  return "";
}

/**
 * @brief Print a human-readable summary from the converter JSON file.
 */
static void printJsonMetadata(const std::string &json_path) {
  std::string content = readFile(json_path);
  if (content.empty()) {
    std::cout << "[nntr_runner] (JSON file empty or unreadable)" << std::endl;
    return;
  }

  std::cout << "[nntr_runner] Model metadata from JSON:" << std::endl;

  auto print = [&](const std::string &label, const std::string &key) {
    std::string val = jsonExtract(content, key);
    if (!val.empty())
      std::cout << "  " << label << ": " << val << std::endl;
  };

  print("model_type",       "model_type");
  print("arch_type",        "arch_type");
  print("hidden_size",      "hidden_size");
  print("num_layers",       "num_layers");
  print("num_heads",        "num_heads");
  print("num_kv_heads",     "num_kv_heads");
  print("head_dim",         "head_dim");
  print("intermediate_size","intermediate_size");
  print("vocab_size",       "vocab_size");
  print("tie_word_emb",     "tie_word_embeddings");

  // Count layers
  size_t layer_count = 0;
  size_t pos = 0;
  while ((pos = content.find("\"layer_type\"", pos)) != std::string::npos) {
    ++layer_count;
    ++pos;
  }
  if (layer_count)
    std::cout << "  total_layers: " << layer_count << std::endl;
}

// ---------------------------------------------------------------------------
// C++ header info printer
// ---------------------------------------------------------------------------

/**
 * @brief Extract and print the generated class name from the C++ header.
 */
static void printHeaderInfo(const std::string &header_path) {
  std::string content = readFile(header_path);
  if (content.empty())
    return;

  // Find the first "class <Name>" declaration
  std::regex re(R"(\bclass\s+(\w+)\b)");
  std::smatch m;
  if (std::regex_search(content, m, re)) {
    std::cout << "[nntr_runner] C++ class: " << m[1].str() << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Layer registration
// ---------------------------------------------------------------------------

/**
 * @brief Register all CausalLM custom layers into the NNTrainer AppContext.
 *
 * Must be called before loadFromConfig so that INI layer types such as
 * "rms_norm", "mha_core", "swiglu", etc. are recognized.
 */
static void registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto *app_context = static_cast<nntrainer::AppContext *>(
    ct_engine.getRegisteredContext("cpu"));

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
    nntrainer::createLayer<causallm::TiedEmbeddingLayer>);
}

// ---------------------------------------------------------------------------
// Model runner
// ---------------------------------------------------------------------------

/**
 * @brief Load, compile, initialize, and summarize a NNTrainer model from INI.
 *
 * @param ini_path   Path to the NNTrainer INI configuration file.
 * @param bin_path   Path to the weight binary file (empty = skip weight load).
 * @return int       0 on success, 1 on failure.
 */
static int runModel(const std::string &ini_path,
                    const std::string &bin_path) {
  std::cout << "[nntr_runner] Loading config: " << ini_path << std::endl;

  ModelHandle model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  if (model->loadFromConfig(ini_path) != 0) {
    std::cerr << "[nntr_runner] ERROR: Failed to load config: " << ini_path
              << std::endl;
    return 1;
  }

  if (model->compile(ml::train::ExecutionMode::INFERENCE) != 0) {
    std::cerr << "[nntr_runner] ERROR: Model compilation failed." << std::endl;
    return 1;
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE) != 0) {
    std::cerr << "[nntr_runner] ERROR: Model initialization failed."
              << std::endl;
    return 1;
  }

  if (!bin_path.empty()) {
    std::cout << "[nntr_runner] Loading weights: " << bin_path << std::endl;
    try {
      model->load(bin_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    } catch (const std::exception &e) {
      std::cerr << "[nntr_runner] ERROR: Weight load failed: " << e.what()
                << std::endl;
      return 1;
    }
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
  std::cout << "[nntr_runner] Model initialized successfully." << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
  std::string ini;
  std::string json;
  std::string header;
  std::string weights;
};

static void printUsage(const char *prog) {
  std::cerr << "Usage:\n"
            << "  " << prog
            << " [--ini model.ini] [--json model.json]"
               " [--header model.h] [--weights model.bin]\n"
            << "  " << prog << " model.ini [model.bin]  (positional)\n"
            << "  " << prog
            << " model.ini model.json model.h model.bin  (auto-detect)\n"
            << "\n"
            << "  --ini      NNTrainer INI config produced by converter.py"
               " (required)\n"
            << "  --json     Converter JSON metadata (optional, printed)\n"
            << "  --header   Auto-generated C++ header (optional, info only)\n"
            << "  --weights  Pre-converted weight binary (optional)\n";
}

/**
 * @brief Parse command-line arguments, supporting both named flags and
 *        positional / auto-detect forms.
 */
static Args parseArgs(int argc, char *argv[]) {
  Args args;

  // Named-flag form: --ini / --json / --header / --weights
  std::map<std::string, std::string *> flags = {
    {"--ini",     &args.ini},
    {"--json",    &args.json},
    {"--header",  &args.header},
    {"--weights", &args.weights},
  };

  std::vector<std::string> positional;
  bool any_flag = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto it = flags.find(a);
    if (it != flags.end()) {
      if (i + 1 >= argc)
        throw std::invalid_argument("Missing value for " + a);
      *it->second = argv[++i];
      any_flag = true;
    } else {
      positional.push_back(a);
    }
  }

  if (!any_flag) {
    // Auto-detect from positional arguments by extension
    for (const auto &p : positional) {
      if (args.ini.empty() && endsWith(p, ".ini"))       args.ini     = p;
      else if (args.json.empty() && endsWith(p, ".json")) args.json   = p;
      else if (args.header.empty() &&
               (endsWith(p, ".h") || endsWith(p, ".hpp"))) args.header = p;
      else if (args.weights.empty() &&
               (endsWith(p, ".bin") ||
                endsWith(p, ".safetensors")))           args.weights  = p;
    }
    // Backward-compatible positional: runner model.ini [model.bin]
    if (args.ini.empty() && !positional.empty())
      args.ini = positional[0];
    if (args.weights.empty() && positional.size() >= 2 &&
        args.json.empty() && args.header.empty())
      args.weights = positional[1];
  }

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
    std::cerr << "[nntr_runner] Argument error: " << e.what() << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  if (args.ini.empty()) {
    std::cerr << "[nntr_runner] ERROR: No INI file specified." << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  try {
    registerCustomLayers();

    // Print supplementary metadata before loading
    if (!args.header.empty())
      printHeaderInfo(args.header);
    if (!args.json.empty())
      printJsonMetadata(args.json);

    return runModel(args.ini, args.weights);
  } catch (const std::exception &e) {
    std::cerr << "[nntr_runner] FATAL: " << e.what() << std::endl;
    return 1;
  }
}
