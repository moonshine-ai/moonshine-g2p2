// CLI mirror of moonshine_onnx_g2p.py (no eSpeak).
#include "moonshine_g2p/cmudict_tsv.hpp"
#include "moonshine_g2p/g2p_word_log.hpp"
#include "moonshine_g2p/moonshine_onnx_g2p.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

using namespace moonshine_g2p;

namespace {

void usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [-d dict.tsv] [--heteronym-onnx PATH] [--oov-onnx PATH] [--cuda] "
         "[--heteronym-optional] [--oov-optional] [--debug-heteronym] "
         "[--log-words] "
         "text tokens...\n"
      << "  Default ONNX paths (if flags omitted): "
         "<dict-dir>/heteronym/model.onnx and "
         "<dict-dir>/oov/model.onnx\n"
      << "  By default both models are required unless --heteronym-optional / "
         "--oov-optional is set.\n"
      << "  --debug-heteronym sets MOONSHINE_G2P_DEBUG_HET=1 (stderr heteronym "
         "ONNX trace).\n";
}

} // namespace

int main(int argc, char **argv) {
  std::optional<std::filesystem::path> dict_path_arg;
  std::optional<std::filesystem::path> het_onnx_arg;
  std::optional<std::filesystem::path> oov_onnx_arg;
  bool use_cuda = false;
  bool log_words = false;
  std::string language = "en_us";
  std::filesystem::path model_root = std::filesystem::path("models");
  std::vector<std::string> text_parts;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--dict" && i + 1 < argc) {
      dict_path_arg = argv[++i];
    } else if (a == "--heteronym-onnx" && i + 1 < argc) {
      het_onnx_arg = argv[++i];
    } else if (a == "--oov-onnx" && i + 1 < argc) {
      oov_onnx_arg = argv[++i];
    } else if (a == "--cuda") {
      use_cuda = true;
    } else if (a == "--log-words" || a == "-v") {
      log_words = true;
    } else if (a == "--debug-heteronym") {
      if (setenv("MOONSHINE_G2P_DEBUG_HET", "1", 1) != 0) {
        std::cerr << "error: setenv MOONSHINE_G2P_DEBUG_HET failed\n";
        return 1;
      }
    } else if (a == "--language") {
      language = argv[++i];
    } else if (a == "--model-root") {
      model_root = argv[++i];
    } else {
      text_parts.push_back(a);
    }
  }

  const std::filesystem::path data_root = model_root / language;
  const std::filesystem::path g2p_config_path = data_root / "g2p-config.json";
  std::optional<std::filesystem::path> dict_path;
  std::optional<std::filesystem::path> het_onnx_path;
  std::optional<std::filesystem::path> oov_onnx_path;
  const bool g2p_config_exists = std::filesystem::exists(g2p_config_path);
  if (!g2p_config_exists) {
    if (!het_onnx_arg && !oov_onnx_arg && !dict_path_arg) {
      std::cerr << "error: g2p-config.json required at "
                << g2p_config_path.generic_string() << "\n";
      usage(argv[0]);
      return 1;
    }
    if (dict_path_arg) {
      dict_path = dict_path_arg;
    }
    if (het_onnx_arg) {
      het_onnx_path = het_onnx_arg;
    }
    if (oov_onnx_arg) {
      oov_onnx_path = oov_onnx_arg;
    }
  } else {
    std::ifstream g2p_config_in(g2p_config_path);
    const nlohmann::json g2p_config = nlohmann::json::parse(g2p_config_in);
    fprintf(stderr, "g2p_config: %s\n", g2p_config.dump(4).c_str());
    if (g2p_config["uses_dictionary"]) {
      dict_path = data_root / "dict_filtered_heteronyms.tsv";
    }
    if (g2p_config["uses_heteronym_model"]) {
      het_onnx_path = data_root / "heteronym" / "model.onnx";
    }
    if (g2p_config["uses_oov_model"]) {
      oov_onnx_path = data_root / "oov" / "model.onnx";
    }
  }

  if (dict_path && !std::filesystem::exists(*dict_path)) {
    std::cerr << "error: dictionary TSV not found at "
              << dict_path->generic_string() << "\n";
    usage(argv[0]);
    return 1;
  }
  if (het_onnx_path && !std::filesystem::exists(*het_onnx_path)) {
    std::cerr << "error: heteronym ONNX model not found at "
              << het_onnx_path->generic_string() << "\n";
    usage(argv[0]);
    return 1;
  }
  if (oov_onnx_path && !std::filesystem::exists(*oov_onnx_path)) {
    std::cerr << "error: OOV ONNX model not found at "
              << oov_onnx_path->generic_string() << "\n";
    usage(argv[0]);
    return 1;
  }

  std::string phrase = "Hello world!";
  if (!text_parts.empty()) {
    phrase.clear();
    for (size_t t = 0; t < text_parts.size(); ++t) {
      if (t > 0) {
        phrase += ' ';
      }
      phrase += text_parts[t];
    }
  }

  try {
    std::unique_ptr<MoonshineOnnxG2p> g2p = std::make_unique<MoonshineOnnxG2p>(
        dict_path, het_onnx_path, oov_onnx_path, use_cuda);
    std::vector<G2pWordLog> word_log;
    std::cout << g2p->text_to_ipa(phrase, log_words ? &word_log : nullptr)
              << '\n';
    if (log_words) {
      for (const auto &e : word_log) {
        std::cerr << format_g2p_word_log_line(e) << '\n';
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
