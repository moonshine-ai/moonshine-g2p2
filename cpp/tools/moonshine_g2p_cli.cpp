// Unified G2P CLI: rule-based dialects (e.g. Spanish) or ONNX bundles under models/<dialect>/.
#include "moonshine_g2p/g2p_word_log.hpp"
#include "moonshine_g2p/moonshine_g2p.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using moonshine_g2p::MoonshineG2P;
using moonshine_g2p::MoonshineG2POptions;
using moonshine_g2p::dialect_resolves_to_french_rules;
using moonshine_g2p::dialect_resolves_to_german_rules;
using moonshine_g2p::dialect_resolves_to_spanish_rules;
using moonshine_g2p::format_g2p_word_log_line;
using moonshine_g2p::spanish_dialect_cli_ids;

namespace {

void usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--language ID] [--model-root DIR]\n"
      << "       [--dict PATH] [--heteronym-onnx PATH] [--oov-onnx PATH]\n"
      << "       [--cuda] [--log-words|-v] [--debug-heteronym]\n"
      << "       [--no-stress] [--broad-phonemes] [--stdin]\n"
      << "       [--german-dict PATH] [--german-syllable-initial-stress]\n"
      << "       [--french-dict PATH] [--french-csv-dir DIR]\n"
      << "       [--no-french-liaison] [--no-french-oov] [--no-french-expand-digits]\n"
      << "       [--no-french-optional-liaison]\n"
      << "       [--print-spanish-dialects] [TEXT...]\n"
      << "  Default dialect: en_us (ONNX under <model-root>/en_us/). Default phrase when no TEXT: "
         "\"Hello world!\".\n"
      << "  Spanish dialects use rule-based G2P (no ONNX). With a Spanish dialect and no TEXT, "
         "stdin is read unless you pass --stdin explicitly for empty input.\n"
      << "  German (de, de-DE, german): rule-based G2P with <model-root>/de/dict.tsv by default; "
         "override with --german-dict.\n"
      << "  French (fr, fr-FR, french): rule-based G2P; default lexicon <model-root>/../data/fr/dict.tsv "
         "or <model-root>/fr/dict.tsv; POS CSVs in the same directory tree.\n"
      << "  -d PATH is an alias for --dict PATH.\n";
}

std::string read_all_stdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

}  // namespace

int main(int argc, char **argv) {
  std::string dialect_str = "en_us";
  MoonshineG2POptions opt;
  bool log_words = false;
  bool force_stdin = false;
  bool print_spanish_dialects = false;
  std::vector<std::string> text_parts;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--print-spanish-dialects") {
      print_spanish_dialects = true;
    } else if ((a == "--dict" || a == "-d") && i + 1 < argc) {
      opt.dict_path_override = argv[++i];
    } else if (a == "--heteronym-onnx" && i + 1 < argc) {
      opt.heteronym_onnx_override = argv[++i];
    } else if (a == "--oov-onnx" && i + 1 < argc) {
      opt.oov_onnx_override = argv[++i];
    } else if (a == "--cuda") {
      opt.use_cuda = true;
    } else if (a == "--log-words" || a == "-v") {
      log_words = true;
    } else if (a == "--debug-heteronym") {
      if (setenv("MOONSHINE_G2P_DEBUG_HET", "1", 1) != 0) {
        std::cerr << "error: setenv MOONSHINE_G2P_DEBUG_HET failed\n";
        return 1;
      }
    } else if (a == "--language" && i + 1 < argc) {
      dialect_str = argv[++i];
    } else if (a == "--model-root" && i + 1 < argc) {
      opt.model_root = argv[++i];
    } else if (a == "--german-dict" && i + 1 < argc) {
      opt.german_dict_path = argv[++i];
    } else if (a == "--french-dict" && i + 1 < argc) {
      opt.french_dict_path = argv[++i];
    } else if (a == "--french-csv-dir" && i + 1 < argc) {
      opt.french_csv_dir = argv[++i];
    } else if (a == "--no-french-liaison") {
      opt.french_liaison = false;
    } else if (a == "--no-french-oov") {
      opt.french_oov_rules = false;
    } else if (a == "--no-french-expand-digits") {
      opt.french_expand_cardinal_digits = false;
    } else if (a == "--no-french-optional-liaison") {
      opt.french_liaison_optional = false;
    } else if (a == "--german-syllable-initial-stress") {
      opt.german_vocoder_stress = false;
    } else if (a == "--no-stress") {
      opt.spanish_with_stress = false;
      opt.german_with_stress = false;
      opt.french_with_stress = false;
    } else if (a == "--broad-phonemes") {
      opt.spanish_narrow_obstruents = false;
    } else if (a == "--stdin") {
      force_stdin = true;
    } else {
      text_parts.push_back(a);
    }
  }

  if (print_spanish_dialects) {
    for (const std::string &id : spanish_dialect_cli_ids()) {
      std::cout << id << '\n';
    }
    if (text_parts.empty() && !force_stdin) {
      return 0;
    }
  }

  std::string phrase;
  if (!text_parts.empty()) {
    for (size_t t = 0; t < text_parts.size(); ++t) {
      if (t > 0) {
        phrase += ' ';
      }
      phrase += text_parts[t];
    }
  } else if (force_stdin ||
             dialect_resolves_to_spanish_rules(dialect_str, opt.spanish_narrow_obstruents) ||
             dialect_resolves_to_german_rules(dialect_str) ||
             dialect_resolves_to_french_rules(dialect_str)) {
    phrase = read_all_stdin();
  } else {
    phrase = "Hello world!";
  }

  try {
    MoonshineG2P g2p(dialect_str, opt);
    std::vector<moonshine_g2p::G2pWordLog> word_log;
    std::cout << g2p.text_to_ipa(phrase, log_words ? &word_log : nullptr) << '\n';
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
