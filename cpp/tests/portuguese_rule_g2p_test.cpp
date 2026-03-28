#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine_g2p/lang-specific/portuguese.hpp"
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
      std::filesystem::temp_directory_path() / ("moonshine_pt_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

std::string python_ipa_from_file(const std::filesystem::path& utf8_file, bool portugal) {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  std::vector<std::string> after;
  if (portugal) {
    after.push_back("--portugal");
  }
  return r::python_ref_from_utf8_file(repo, "portuguese_g2p_ref.py", utf8_file, after);
}

std::string python_ipa_one_line(const std::string& line, bool portugal) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path tmp =
      std::filesystem::temp_directory_path() / ("pt_g2p_line_" + std::to_string(tick) + ".txt");
  {
    std::ofstream o(tmp, std::ios::binary);
    o << line;
  }
  const std::string py = python_ipa_from_file(tmp, portugal);
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  return py;
}

std::vector<std::string> python_ipa_first_lines(const std::filesystem::path& text_file, int n, bool portugal) {
  std::vector<std::string> after;
  if (portugal) {
    after.push_back("--portugal");
  }
  return r::python_ref_first_lines(r::repo_root_from_tests_cpp(__FILE__), "portuguese_g2p_ref.py", text_file, n,
                                   {}, after);
}

bool python_portuguese_import_ok() {
  return r::python_import_ok(r::repo_root_from_tests_cpp(__FILE__), "from portuguese_rule_g2p import text_to_ipa");
}

}  // namespace

TEST_CASE("portuguese: dialect flags") {
  using moonshine_g2p::dialect_resolves_to_brazilian_portuguese_rules;
  using moonshine_g2p::dialect_resolves_to_portugal_rules;
  CHECK(dialect_resolves_to_brazilian_portuguese_rules("pt-BR"));
  CHECK(dialect_resolves_to_brazilian_portuguese_rules("pt_br"));
  CHECK(dialect_resolves_to_portugal_rules("pt-PT"));
  CHECK(dialect_resolves_to_portugal_rules("pt_pt"));
  CHECK(dialect_resolves_to_portugal_rules("portugal"));
  CHECK_FALSE(dialect_resolves_to_portugal_rules("pt-br"));
}

TEST_CASE("portuguese: lowercase homograph overrides capitalized") {
  const auto p = make_temp_tsv(
      "Rio\twrong\n"
      "rio\tright\n");
  moonshine_g2p::PortugueseRuleG2p g(p, false);
  CHECK(g.word_to_ipa("rio") == "right");
  std::filesystem::remove(p);
}

TEST_CASE("portuguese: lexicon stress not shifted by vocoder") {
  static const std::string kPri{"\xCB\x88"};
  const auto p = make_temp_tsv("testlex\tt\xC9\xAA\xCB\x88st\n");
  moonshine_g2p::PortugueseRuleG2p g(p, false);
  const std::string ipa = g.word_to_ipa("testlex");
  CHECK(ipa.find(kPri) != std::string::npos);
  std::filesystem::remove(p);
}

TEST_CASE("portuguese: casa matches Python when data and python3 exist") {
  const std::filesystem::path dict = r::repo_root_from_tests_cpp(__FILE__) / "data" / "pt_br" / "dict.tsv";
  if (!std::filesystem::is_regular_file(dict) || !python_portuguese_import_ok()) {
    return;
  }
  moonshine_g2p::PortugueseRuleG2p g(dict, false);
  const std::string py = python_ipa_one_line("casa", false);
  CHECK(g.text_to_ipa("casa") == py);
}

TEST_CASE("portuguese: wiki-text first 100 lines pt_br match Python when data present") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = repo / "data" / "pt_br" / "dict.tsv";
  const std::filesystem::path wiki = repo / "data" / "pt_br" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !python_portuguese_import_ok()) {
    return;
  }
  moonshine_g2p::PortugueseRuleG2p g(dict, false);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = python_ipa_first_lines(wiki, static_cast<int>(src.size()), false);
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("pt_br wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}

TEST_CASE("portuguese: wiki-text first 100 lines pt_pt match Python when data present") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = repo / "data" / "pt_pt" / "dict.tsv";
  const std::filesystem::path wiki = repo / "data" / "pt_pt" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !python_portuguese_import_ok()) {
    return;
  }
  moonshine_g2p::PortugueseRuleG2p g(dict, true);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = python_ipa_first_lines(wiki, static_cast<int>(src.size()), true);
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("pt_pt wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
