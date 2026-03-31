// SPDX-License-Identifier: Apache-2.0
/**
 * @file   nntr_runner.cpp
 * @date   31 Mar 2026
 * @brief  Generic NNTrainer model runner for TorchFXConverter output.
 *
 * Works like nntr_causallm but is architecture-agnostic: instead of
 * compiling a per-model C++ class, it loads the NNTrainer INI produced
 * by converter.py via loadFromConfig(), then runs one inference pass
 * with a zero-filled dummy input to verify the model end-to-end.
 *
 * Primary usage — pass the converter output directory directly:
 *
 *   nntr_runner ./out/qwen3_05b/
 *
 * The runner auto-discovers files inside the directory:
 *   *.ini              → NNTrainer config  (required)
 *   *.json             → converter metadata (printed before load)
 *   *.h / *.hpp        → C++ header        (class name printed)
 *   *.bin / *.safetensors → weight binary  (loaded after init)
 *
 * Explicit file flags (any subset, override auto-discovery):
 *   nntr_runner --ini m.ini --json m.json --header m.h --weights m.bin
 *
 * Backward-compatible positional form:
 *   nntr_runner model.ini
 *   nntr_runner model.ini model.bin
 *
 * Exit codes:
 *   0 — model initialized and inference pass succeeded
 *   1 — error (details on stderr)
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
  std::string header;
  std::string weights;
};

/**
 * @brief Scan a directory and assign files to roles by extension.
 *
 * Picks the first file matching each extension.  When multiple INI files
 * exist, the one whose basename most closely matches the directory name is
 * preferred.
 */
static FileSet discoverFiles(const std::string &dir) {
  FileSet fs;
  DIR *d = opendir(dir.c_str());
  if (!d)
    throw std::runtime_error("Cannot open directory: " + dir);

  std::string base = dir;
  if (!base.empty() && base.back() == '/')
    base.pop_back();
  size_t slash = base.rfind('/');
  if (slash != std::string::npos)
    base = base.substr(slash + 1);

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
    else if (fs.header.empty() &&
             (endsWith(name, ".h") || endsWith(name, ".hpp")))
      fs.header = full;
    else if (fs.weights.empty() &&
             (endsWith(name, ".bin") || endsWith(name, ".safetensors")))
      fs.weights = full;
  }
  closedir(d);
  return fs;
}

// ---------------------------------------------------------------------------
// JSON metadata printer (no external dependency)
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
  std::string content = readFile(json_path);
  if (content.empty())
    return;

  std::cout << "[nntr_runner] Model metadata:" << std::endl;
  auto p = [&](const std::string &label, const std::string &key) {
    std::string v = jsonExtract(content, key);
    if (!v.empty())
      std::cout << "  " << label << ": " << v << std::endl;
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
  p("tie_word_emb",      "tie_word_embeddings");

  size_t n = 0, pos = 0;
  while ((pos = content.find("\"layer_type\"", pos)) != std::string::npos) {
    ++n; ++pos;
  }
  if (n)
    std::cout << "  total_layers: " << n << std::endl;
}

// ---------------------------------------------------------------------------
// C++ header info printer
// ---------------------------------------------------------------------------

static void printHeaderInfo(const std::string &header_path) {
  std::string content = readFile(header_path);
  if (content.empty())
    return;
  std::regex re(R"(\bclass\s+(\w+)\b)");
  std::smatch m;
  if (std::regex_search(content, m, re))
    std::cout << "[nntr_runner] C++ class: " << m[1].str() << std::endl;
}

// ---------------------------------------------------------------------------
// Custom layer registration
// ---------------------------------------------------------------------------

/**
 * @brief Register all CausalLM custom layers into the NNTrainer AppContext.
 *
 * Called once before loadFromConfig so that INI layer types such as
 * "rms_norm", "mha_core", "swiglu", etc. are resolved correctly.
 */
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
    // Layer already registered — safe to ignore on repeated calls.
    std::cerr << "[nntr_runner] registerFactory warning: " << e.what()
              << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Model run
// ---------------------------------------------------------------------------

/**
 * @brief Load model from INI, compile, initialize, optionally load weights,
 *        then run one inference pass with a zero-filled dummy input.
 *
 * The dummy input uses the shape reported by model->getInputDimension()
 * so it works for any architecture produced by converter.py.
 *
 * @param ini_path   Path to the NNTrainer INI file.
 * @param bin_path   Path to the weight binary (empty = skip).
 * @return int       0 on success, 1 on failure.
 */
static int runModel(const std::string &ini_path,
                    const std::string &bin_path) {
  std::cout << "[nntr_runner] Loading config: " << ini_path << std::endl;

  ModelHandle model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  if (model->loadFromConfig(ini_path) != 0) {
    std::cerr << "[nntr_runner] ERROR: loadFromConfig failed: " << ini_path
              << std::endl;
    return 1;
  }

  if (model->compile(ml::train::ExecutionMode::INFERENCE) != 0) {
    std::cerr << "[nntr_runner] ERROR: compile failed." << std::endl;
    return 1;
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE) != 0) {
    std::cerr << "[nntr_runner] ERROR: initialize failed." << std::endl;
    return 1;
  }

  if (!bin_path.empty()) {
    std::cout << "[nntr_runner] Loading weights: " << bin_path << std::endl;
    try {
      model->load(bin_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    } catch (const std::exception &e) {
      std::cerr << "[nntr_runner] ERROR: weight load failed: " << e.what()
                << std::endl;
      return 1;
    }
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  // ----- dummy inference pass -----
  // Build zero-filled input buffers shaped after each model input.
  auto input_dims = model->getInputDimension();
  if (input_dims.empty()) {
    std::cerr << "[nntr_runner] WARNING: could not query input dimensions; "
                 "skipping inference pass."
              << std::endl;
  } else {
    std::cout << "[nntr_runner] Running dummy inference (zero input)..."
              << std::endl;

    std::vector<std::vector<float>> buffers;
    std::vector<float *> input_ptrs;
    for (const auto &dim : input_dims) {
      size_t sz = dim.getDataLen();
      buffers.emplace_back(sz, 0.0f);
      input_ptrs.push_back(buffers.back().data());
    }

    try {
      auto outputs = model->inference(1, input_ptrs);
      std::cout << "[nntr_runner] Inference OK — " << outputs.size()
                << " output tensor(s)." << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "[nntr_runner] ERROR: inference failed: " << e.what()
                << std::endl;
      return 1;
    }
  }

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
  std::cerr
    << "Usage:\n"
    << "  " << prog << " <output_dir/>         (auto-discover from directory)\n"
    << "  " << prog
    << " [--ini m.ini] [--json m.json] [--header m.h] [--weights m.bin]\n"
    << "  " << prog << " model.ini [model.bin]  (positional, backward compat)\n"
    << "\n"
    << "  output_dir   Directory produced by converter.py --format all\n"
    << "  --ini        NNTrainer INI config   (required)\n"
    << "  --json       Converter JSON metadata (optional, printed)\n"
    << "  --header     Auto-generated C++ header (optional, info only)\n"
    << "  --weights    Pre-converted weight binary (optional)\n";
}

static Args parseArgs(int argc, char *argv[]) {
  Args args;

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
        throw std::invalid_argument("Missing value after " + a);
      *it->second = argv[++i];
      any_flag = true;
    } else {
      positional.push_back(a);
    }
  }

  if (!any_flag) {
    if (positional.size() == 1 && isDirectory(positional[0])) {
      // Primary form: single directory argument
      FileSet fs = discoverFiles(positional[0]);
      args.ini     = fs.ini;
      args.json    = fs.json;
      args.header  = fs.header;
      args.weights = fs.weights;
    } else {
      // Auto-detect from positional args by extension
      for (const auto &p : positional) {
        if      (args.ini.empty()     && endsWith(p, ".ini"))          args.ini     = p;
        else if (args.json.empty()    && endsWith(p, ".json"))         args.json    = p;
        else if (args.header.empty()  && (endsWith(p, ".h") ||
                                          endsWith(p, ".hpp")))        args.header  = p;
        else if (args.weights.empty() && (endsWith(p, ".bin") ||
                                          endsWith(p, ".safetensors"))) args.weights = p;
      }
      // Fallback: first positional = ini, second (if no json/header) = weights
      if (args.ini.empty() && !positional.empty())
        args.ini = positional[0];
      if (args.weights.empty() && positional.size() >= 2 &&
          args.json.empty() && args.header.empty())
        args.weights = positional[1];
    }
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
    std::cerr << "[nntr_runner] ERROR: No INI file found or specified."
              << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  try {
    registerCustomLayers();

    if (!args.header.empty()) printHeaderInfo(args.header);
    if (!args.json.empty())   printJsonMetadata(args.json);

    return runModel(args.ini, args.weights);
  } catch (const std::exception &e) {
    std::cerr << "[nntr_runner] FATAL: " << e.what() << std::endl;
    return 1;
  }
}
