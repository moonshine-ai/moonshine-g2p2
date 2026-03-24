// CLI mirror of moonshine_onnx_g2p.py (no eSpeak).
#include "moonshine_g2p/cmudict_tsv.hpp"
#include "moonshine_g2p/g2p_word_log.hpp"
#include "moonshine_g2p/moonshine_onnx_g2p.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace moonshine_g2p;

namespace {

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [-d dict.tsv] [--heteronym-onnx PATH] [--oov-onnx PATH] [--cuda] "
               "[--heteronym-optional] [--oov-optional] [--debug-heteronym] [--log-words] "
               "text tokens...\n"
            << "  Default ONNX paths (if flags omitted): <dict-dir>/heteronym/model.onnx and "
               "<dict-dir>/oov/model.onnx\n"
            << "  By default both models are required unless --heteronym-optional / "
               "--oov-optional is set.\n"
            << "  --debug-heteronym sets MOONSHINE_G2P_DEBUG_HET=1 (stderr heteronym ONNX trace).\n";
}

/// If the file exists, return it; if *require_model* is true and the file is missing, return the
/// candidate so MoonshineOnnxG2p can throw with that path; otherwise nullopt.
std::optional<std::filesystem::path> resolve_onnx_path(const std::filesystem::path& candidate,
                                                       bool require_model) {
  if (std::filesystem::is_regular_file(candidate)) {
    return candidate;
  }
  if (require_model) {
    return candidate;
  }
  return std::nullopt;
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<std::filesystem::path> dict_path;
  std::optional<std::filesystem::path> het_onnx_arg;
  std::optional<std::filesystem::path> oov_onnx_arg;
  bool use_cuda = false;
  bool log_words = false;
  bool require_heteronym_model = true;
  bool require_oov_model = true;
  std::vector<std::string> text_parts;

#ifdef MOONSHINE_G2P_DEFAULT_DICT
  dict_path = std::filesystem::path(MOONSHINE_G2P_DEFAULT_DICT);
#endif

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "-d" && i + 1 < argc) {
      dict_path = argv[++i];
    } else if (a == "--heteronym-onnx" && i + 1 < argc) {
      het_onnx_arg = argv[++i];
    } else if (a == "--oov-onnx" && i + 1 < argc) {
      oov_onnx_arg = argv[++i];
    } else if (a == "--cuda") {
      use_cuda = true;
    } else if (a == "--log-words" || a == "-v") {
      log_words = true;
    } else if (a == "--heteronym-optional") {
      require_heteronym_model = false;
    } else if (a == "--oov-optional") {
      require_oov_model = false;
    } else if (a == "--debug-heteronym") {
      if (setenv("MOONSHINE_G2P_DEBUG_HET", "1", 1) != 0) {
        std::cerr << "error: setenv MOONSHINE_G2P_DEBUG_HET failed\n";
        return 1;
      }
    } else {
      text_parts.push_back(a);
    }
  }

  if (!dict_path || !std::filesystem::exists(*dict_path)) {
    std::cerr << "error: dictionary TSV required (-d path)\n";
    usage(argv[0]);
    return 1;
  }

  const std::filesystem::path dict_parent = dict_path->parent_path();
  const std::filesystem::path het_candidate =
      het_onnx_arg.value_or(dict_parent / "heteronym" / "model.onnx");
  const std::filesystem::path oov_candidate =
      oov_onnx_arg.value_or(dict_parent / "oov" / "model.onnx");

  std::cerr << "moonshine_g2p: heteronym ONNX path: "
            << std::filesystem::absolute(het_candidate).generic_string() << '\n';
  std::cerr << "moonshine_g2p: OOV ONNX path: "
            << std::filesystem::absolute(oov_candidate).generic_string() << '\n';

  const std::optional<std::filesystem::path> het_resolved =
      resolve_onnx_path(het_candidate, require_heteronym_model);
  const std::optional<std::filesystem::path> oov_resolved =
      resolve_onnx_path(oov_candidate, require_oov_model);

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
    CmudictTsv dict(*dict_path);
    MoonshineOnnxG2p g2p(dict, het_resolved, oov_resolved, use_cuda, require_heteronym_model,
                         require_oov_model);
    std::vector<G2pWordLog> word_log;
    std::cout << g2p.text_to_ipa(phrase, log_words ? &word_log : nullptr) << '\n';
    if (log_words) {
      for (const auto& e : word_log) {
        std::cerr << format_g2p_word_log_line(e) << '\n';
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
