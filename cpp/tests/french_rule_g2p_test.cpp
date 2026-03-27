#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine_g2p/lang-specific/french.hpp"

#include <filesystem>
#include <string>

namespace {

std::filesystem::path repo_root() {
  return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
}

std::string strip_stress(std::string s) {
  static const std::string kPri{"\xCB\x88"};
  static const std::string kSec{"\xCB\x8C"};
  for (;;) {
    const size_t p = s.find(kPri);
    if (p == std::string::npos) {
      break;
    }
    s.erase(p, kPri.size());
  }
  for (;;) {
    const size_t p = s.find(kSec);
    if (p == std::string::npos) {
      break;
    }
    s.erase(p, kSec.size());
  }
  return s;
}

bool french_dict_present() {
  return std::filesystem::is_regular_file(repo_root() / "data" / "fr" / "dict.tsv");
}

}  // namespace

TEST_CASE("french: dialect_resolves_to_french_rules") {
  using moonshine_g2p::dialect_resolves_to_french_rules;
  CHECK(dialect_resolves_to_french_rules("fr"));
  CHECK(dialect_resolves_to_french_rules("fr-FR"));
  CHECK(dialect_resolves_to_french_rules("FR_fr"));
  CHECK(dialect_resolves_to_french_rules("french"));
  CHECK_FALSE(dialect_resolves_to_french_rules("de"));
}

TEST_CASE("french: ensure_french_nuclear_stress") {
  using moonshine_g2p::FrenchRuleG2p;
  CHECK(FrenchRuleG2p::ensure_french_nuclear_stress("bɔ̃ʒuʁ") == "bɔ̃ʒˈuʁ");
  CHECK(FrenchRuleG2p::ensure_french_nuclear_stress("kɔmɑ̃") == "kɔmˈɑ̃");
}

TEST_CASE("french: liaison les amis" * doctest::skip(!french_dict_present())) {
  const auto dict = repo_root() / "data" / "fr" / "dict.tsv";
  const auto csv = dict.parent_path();
  moonshine_g2p::FrenchRuleG2p g(dict, csv);
  const std::string out = g.text_to_ipa("les amis");
  const std::string first = strip_stress(out.substr(0, out.find(' ')));
  CHECK(first == "lez");
  CHECK(strip_stress(out).find("ami") != std::string::npos);
}

TEST_CASE("french: En 1891 cardinal expansion" * doctest::skip(!french_dict_present())) {
  const auto dict = repo_root() / "data" / "fr" / "dict.tsv";
  const auto csv = dict.parent_path();
  moonshine_g2p::FrenchRuleG2p g(dict, csv);
  const std::string out = g.text_to_ipa("En 1891");
  const std::string u = strip_stress(out);
  CHECK(u.find("mil") != std::string::npos);
  CHECK(u.find("katʁv") != std::string::npos);
}

TEST_CASE("french: punctuation keeps space before next word" * doctest::skip(!french_dict_present())) {
  const auto dict = repo_root() / "data" / "fr" / "dict.tsv";
  const auto csv = dict.parent_path();
  moonshine_g2p::FrenchRuleG2p g(dict, csv);
  CHECK(g.text_to_ipa("Bonjour! Salut").find("! ") != std::string::npos);
  CHECK(g.text_to_ipa("Vu? Oui").find("? ") != std::string::npos);
}

TEST_CASE("french: hyphenated OOV allez-vous matches Python (UTF-8 trim + stress)" *
          doctest::skip(!french_dict_present())) {
  const auto dict = repo_root() / "data" / "fr" / "dict.tsv";
  const auto csv = dict.parent_path();
  moonshine_g2p::FrenchRuleG2p g(dict, csv);
  CHECK(g.text_to_ipa("comment allez-vous") == "kɔmˈɑ̃ allˈə-vˈu");
}

TEST_CASE("french: uppercase accented letters in words (Saint-Étienne)" *
          doctest::skip(!french_dict_present())) {
  const auto dict = repo_root() / "data" / "fr" / "dict.tsv";
  const auto csv = dict.parent_path();
  moonshine_g2p::FrenchRuleG2p g(dict, csv);
  CHECK(g.text_to_ipa("Saint-\xC3\x89tienne") == "sˈɛ̃-etjˈɛ̃n");
}
