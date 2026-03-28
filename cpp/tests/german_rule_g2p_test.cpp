#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine_g2p/lang-specific/german.hpp"
#include "rule_g2p_test_support.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace r = moonshine_g2p::rule_g2p_test;

namespace {

std::filesystem::path make_temp_tsv(const char* contents) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path p =
      std::filesystem::temp_directory_path() / ("moonshine_german_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

std::vector<std::string> python_ipa_first_lines(const std::filesystem::path& text_file, int n) {
  return r::python_ref_first_lines(r::repo_root_from_tests_cpp(__FILE__), "german_g2p_ref.py", text_file, n);
}

bool python_german_import_ok() {
  return r::python_import_ok(r::repo_root_from_tests_cpp(__FILE__), "from german_rule_g2p import text_to_ipa");
}

}  // namespace

TEST_CASE("german: lowercase homograph overrides capitalized") {
  const auto p = make_temp_tsv(
      "Die\tdaɪ\n"
      "die\tdiː\n");
  moonshine_g2p::GermanRuleG2p g(p);
  CHECK(g.word_to_ipa("die") == "diː");
  std::filesystem::remove(p);
}

TEST_CASE("german: lexicon entry with syllable-initial stress gets vocoder shift") {
  const auto p = make_temp_tsv("komme\tˈkɔmə\n");
  moonshine_g2p::GermanRuleG2p g(p);
  CHECK(g.word_to_ipa("komme") == "kˈɔmə");
  std::filesystem::remove(p);
}

TEST_CASE("german: syllable-initial stress preserved when vocoder_stress false") {
  const auto p = make_temp_tsv("komme\tˈkɔmə\n");
  moonshine_g2p::GermanRuleG2p::Options o;
  o.vocoder_stress = false;
  moonshine_g2p::GermanRuleG2p g(p, o);
  CHECK(g.word_to_ipa("komme") == "ˈkɔmə");
  std::filesystem::remove(p);
}

TEST_CASE("german: OOV rules machen") {
  const auto p = make_temp_tsv("x\ty\n");  // dummy so file parses
  moonshine_g2p::GermanRuleG2p g(p);
  const std::string ipa = g.word_to_ipa("machen");
  static const std::string kPri{"\xCB\x88"};
  CHECK(ipa.find(kPri) != std::string::npos);
  const bool has_ch_or_x =
      ipa.find('x') != std::string::npos || ipa.find("\xC3\xA7") != std::string::npos;
  CHECK(has_ch_or_x);
  std::filesystem::remove(p);
}

TEST_CASE("german: normalize_ipa_stress_for_vocoder idempotent") {
  using moonshine_g2p::GermanRuleG2p;
  std::string a = GermanRuleG2p::normalize_ipa_stress_for_vocoder("ˈkɔmə");
  std::string b = GermanRuleG2p::normalize_ipa_stress_for_vocoder(a);
  CHECK(a == b);
  CHECK(a == "kˈɔmə");
}

TEST_CASE("german: dialect_resolves_to_german_rules") {
  using moonshine_g2p::dialect_resolves_to_german_rules;
  CHECK(dialect_resolves_to_german_rules("de"));
  CHECK(dialect_resolves_to_german_rules("de-DE"));
  CHECK(dialect_resolves_to_german_rules("DE_de"));
  CHECK(dialect_resolves_to_german_rules("german"));
  CHECK_FALSE(dialect_resolves_to_german_rules("en_us"));
}

TEST_CASE("german: text token preserves comma") {
  const auto p = make_temp_tsv("Hallo\thaˈloː\n");
  moonshine_g2p::GermanRuleG2p g(p);
  const std::string t = g.text_to_ipa("Hallo, du");
  CHECK(t.find(',') != std::string::npos);
  std::filesystem::remove(p);
}

TEST_CASE("german: wiki-text first 100 lines match Python when data and python3 exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = repo / "data" / "de" / "dict.tsv";
  const std::filesystem::path wiki = repo / "data" / "de" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !python_german_import_ok()) {
    return;
  }
  moonshine_g2p::GermanRuleG2p g(dict);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = python_ipa_first_lines(wiki, static_cast<int>(src.size()));
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
