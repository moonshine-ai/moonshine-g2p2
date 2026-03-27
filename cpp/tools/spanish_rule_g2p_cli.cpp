// Rule-based Spanish G2P CLI (mirrors spanish_rule_g2p.py text output; no eSpeak line).
#include "moonshine_g2p/lang-specific/spanish.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using moonshine_g2p::SpanishDialect;
using moonshine_g2p::spanish_dialect_cli_ids;
using moonshine_g2p::spanish_dialect_from_cli_id;
using moonshine_g2p::spanish_text_to_ipa;

namespace {

void usage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " [--dialect ID] [--no-stress] [--broad-phonemes] [--stdin] [TEXT...]\n"
            << "  Default dialect: es-MX. Reads stdin when no TEXT and not --stdin, or with "
               "--stdin.\n"
            << "  Dialect IDs: ";
  const auto ids = spanish_dialect_cli_ids();
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) {
      std::cerr << ", ";
    }
    std::cerr << ids[i];
  }
  std::cerr << "\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string dialect_id = "es-MX";
  bool with_stress = true;
  bool narrow_obstruents = true;
  bool force_stdin = false;
  std::vector<std::string> text_parts;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--dialect" && i + 1 < argc) {
      dialect_id = argv[++i];
    } else if (a == "--no-stress") {
      with_stress = false;
    } else if (a == "--broad-phonemes") {
      narrow_obstruents = false;
    } else if (a == "--stdin") {
      force_stdin = true;
    } else {
      text_parts.push_back(a);
    }
  }

  std::string raw;
  if (force_stdin || text_parts.empty()) {
    std::ostringstream oss;
    oss << std::cin.rdbuf();
    raw = oss.str();
  } else {
    for (size_t t = 0; t < text_parts.size(); ++t) {
      if (t) {
        raw.push_back(' ');
      }
      raw += text_parts[t];
    }
  }

  try {
    const SpanishDialect dialect =
        spanish_dialect_from_cli_id(dialect_id, narrow_obstruents);
    const std::string ipa = spanish_text_to_ipa(raw, dialect, with_stress);
    std::cout << ipa << '\n';
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
