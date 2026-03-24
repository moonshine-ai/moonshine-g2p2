// CLI mirror of moonshine_onnx_g2p.py (no eSpeak).
#include "moonshine_g2p/cmudict_tsv.hpp"
#include "moonshine_g2p/moonshine_onnx_g2p.hpp"

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
               "text tokens...\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<std::filesystem::path> dict_path;
  std::optional<std::filesystem::path> het_onnx;
  std::optional<std::filesystem::path> oov_onnx;
  bool use_cuda = false;
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
      het_onnx = argv[++i];
    } else if (a == "--oov-onnx" && i + 1 < argc) {
      oov_onnx = argv[++i];
    } else if (a == "--cuda") {
      use_cuda = true;
    } else {
      text_parts.push_back(a);
    }
  }

  if (!dict_path || !std::filesystem::exists(*dict_path)) {
    std::cerr << "error: dictionary TSV required (-d path)\n";
    usage(argv[0]);
    return 1;
  }

  if (het_onnx && !std::filesystem::exists(*het_onnx)) {
    std::cerr << "error: heteronym onnx not found\n";
    return 1;
  }
  if (oov_onnx && !std::filesystem::exists(*oov_onnx)) {
    std::cerr << "error: oov onnx not found\n";
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
    CmudictTsv dict(*dict_path);
    MoonshineOnnxG2p g2p(dict, het_onnx, oov_onnx, use_cuda);
    std::cout << g2p.text_to_ipa(phrase) << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
