#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/lang-specific/italian.h"
#include "rule-g2p-test-support.h"

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
      std::filesystem::temp_directory_path() / ("moonshine_italian_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

std::string python_ipa_from_file(const std::filesystem::path& utf8_file) {
  return r::python_ref_from_utf8_file(r::repo_root_from_tests_cpp(__FILE__), "italian_g2p_ref.py", utf8_file);
}

std::string python_ipa_one_line(const std::string& line) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path tmp =
      std::filesystem::temp_directory_path() / ("it_g2p_line_" + std::to_string(tick) + ".txt");
  {
    std::ofstream o(tmp, std::ios::binary);
    o << line;
  }
  const std::string py = python_ipa_from_file(tmp);
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  return py;
}

std::vector<std::string> python_ipa_first_lines(const std::filesystem::path& text_file, int n) {
  return r::python_ref_first_lines(r::repo_root_from_tests_cpp(__FILE__), "italian_g2p_ref.py", text_file, n);
}

bool python_italian_import_ok() {
  return r::python_import_ok(r::repo_root_from_tests_cpp(__FILE__), "from italian_rule_g2p import text_to_ipa");
}

}  // namespace

TEST_CASE("italian: dialect_resolves_to_italian_rules") {
  using moonshine_g2p::dialect_resolves_to_italian_rules;
  CHECK(dialect_resolves_to_italian_rules("it"));
  CHECK(dialect_resolves_to_italian_rules("it-IT"));
  CHECK(dialect_resolves_to_italian_rules("IT_it"));
  CHECK(dialect_resolves_to_italian_rules("italian"));
  CHECK_FALSE(dialect_resolves_to_italian_rules("de"));
}

TEST_CASE("italian: lowercase homograph overrides capitalized") {
  const auto p = make_temp_tsv(
      "Roma\twrong\n"
      "roma\tright\n");
  moonshine_g2p::ItalianRuleG2p g(p);
  CHECK(g.word_to_ipa("roma") == "right");
  std::filesystem::remove(p);
}

TEST_CASE("italian: lexicon stress not shifted by vocoder") {
  static const std::string kPri{"\xCB\x88"};
  const auto p = make_temp_tsv("testlex\tt\xC9\xAA\xCB\x88st\n");
  moonshine_g2p::ItalianRuleG2p g(p);
  const std::string ipa = g.word_to_ipa("testlex");
  CHECK(ipa.find(kPri) != std::string::npos);
  std::filesystem::remove(p);
}

TEST_CASE("italian: c'è matches Python when available") {
  const std::filesystem::path dict = r::repo_root_from_tests_cpp(__FILE__) / "data" / "it" / "dict.tsv";
  if (!std::filesystem::is_regular_file(dict) || !python_italian_import_ok()) {
    return;
  }
  moonshine_g2p::ItalianRuleG2p g(dict);
  const std::string py_ascii = python_ipa_one_line("c'è");
  CHECK(g.text_to_ipa("c'è") == py_ascii);
  const std::string py_typo = python_ipa_one_line("c\u2019\u00e8");
  CHECK(g.text_to_ipa("c\u2019\u00e8") == py_typo);
}

TEST_CASE("italian: wiki-text first 100 lines match Python when data and python3 exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = repo / "data" / "it" / "dict.tsv";
  const std::filesystem::path wiki = repo / "data" / "it" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !python_italian_import_ok()) {
    return;
  }
  moonshine_g2p::ItalianRuleG2p g(dict);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = python_ipa_first_lines(wiki, static_cast<int>(src.size()));
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
