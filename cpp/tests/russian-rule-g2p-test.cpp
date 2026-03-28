#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/lang-specific/russian.h"
#ifdef MOONSHINE_G2P_WITH_MOONSHINE_G2P_CLASS
#include "moonshine-g2p/moonshine-g2p.h"
#endif

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
      std::filesystem::temp_directory_path() / ("moonshine_russian_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

std::string python_ipa_from_file(const std::filesystem::path& utf8_file) {
  return r::python_ref_from_utf8_file(r::repo_root_from_tests_cpp(__FILE__), "russian_g2p_ref.py", utf8_file);
}

std::string python_ipa_one_line(const std::string& line) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path tmp =
      std::filesystem::temp_directory_path() / ("ru_g2p_line_" + std::to_string(tick) + ".txt");
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
  return r::python_ref_first_lines(r::repo_root_from_tests_cpp(__FILE__), "russian_g2p_ref.py", text_file, n);
}

bool python_russian_import_ok() {
  return r::python_import_ok(r::repo_root_from_tests_cpp(__FILE__), "from russian_rule_g2p import text_to_ipa");
}

}  // namespace

TEST_CASE("russian: dialect_resolves_to_russian_rules") {
  using moonshine_g2p::dialect_resolves_to_russian_rules;
  CHECK(dialect_resolves_to_russian_rules("ru"));
  CHECK(dialect_resolves_to_russian_rules("ru-RU"));
  CHECK(dialect_resolves_to_russian_rules("RU_ru"));
  CHECK(dialect_resolves_to_russian_rules("russian"));
  CHECK_FALSE(dialect_resolves_to_russian_rules("de"));
}

TEST_CASE("russian: lowercase homograph overrides capitalized") {
  const auto p = make_temp_tsv(
      "\xD0\xA0\xD0\xBE\xD0\xBC\xD0\xB0\twrong\n"
      "\xD1\x80\xD0\xBE\xD0\xBC\xD0\xB0\tright\n");
  moonshine_g2p::RussianRuleG2p g(p);
  CHECK(g.word_to_ipa("\xD1\x80\xD0\xBE\xD0\xBC\xD0\xB0") == "right");
  std::filesystem::remove(p);
}

#ifdef MOONSHINE_G2P_WITH_MOONSHINE_G2P_CLASS
TEST_CASE("russian: MoonshineG2P ru uses rule backend and matches RussianRuleG2p") {
  const auto p = make_temp_tsv(
      "\xD1\x82\xD0\xB5\xD1\x81\xD1\x82\t\xCB\x88t\xC9\x9Bst\n");  // тест + IPA
  moonshine_g2p::MoonshineG2POptions opt;
  opt.russian_dict_path = p;
  moonshine_g2p::MoonshineG2P g("ru", opt);
  CHECK(g.uses_russian_rules());
  CHECK_FALSE(g.uses_onnx());
  CHECK(g.dialect_id() == "ru-RU");
  moonshine_g2p::RussianRuleG2p direct(p);
  const std::string w = "\xD1\x82\xD0\xB5\xD1\x81\xD1\x82";
  CHECK(g.text_to_ipa(w) == direct.text_to_ipa(w));
  std::filesystem::remove(p);
}
#endif

TEST_CASE("russian: litva matches Python when data and python3 exist") {
  const std::filesystem::path dict = r::repo_root_from_tests_cpp(__FILE__) / "data" / "ru" / "dict.tsv";
  if (!std::filesystem::is_regular_file(dict) || !python_russian_import_ok()) {
    return;
  }
  moonshine_g2p::RussianRuleG2p g(dict);
  const std::string py = python_ipa_one_line(
      "\xD0\x9B\xD0\xB8\xD1\x82\xD0\xB2\xD0\xB0");
  CHECK(g.word_to_ipa(
            "\xD0\x9B\xD0\xB8\xD1\x82\xD0\xB2\xD0\xB0") == py);
}

TEST_CASE("russian: Cyrillic preposition plus 1891 matches Python when data and python3 exist") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = repo / "data" / "ru" / "dict.tsv";
  if (!std::filesystem::is_regular_file(dict) || !python_russian_import_ok()) {
    return;
  }
  const std::string line = "\xD0\x92 1891";  // "В 1891"
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path tmp =
      std::filesystem::temp_directory_path() / ("ru_g2p_digits_" + std::to_string(tick) + ".txt");
  {
    std::ofstream o(tmp, std::ios::binary);
    o << line;
  }
  const std::vector<std::string> py = python_ipa_first_lines(tmp, 1);
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  REQUIRE(py.size() == 1);
  moonshine_g2p::RussianRuleG2p g(dict);
  CHECK(g.text_to_ipa(line) == py[0]);
}

TEST_CASE("russian: wiki-text first 100 lines match Python when data and python3 exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = repo / "data" / "ru" / "dict.tsv";
  const std::filesystem::path wiki = repo / "data" / "ru" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !python_russian_import_ok()) {
    return;
  }
  moonshine_g2p::RussianRuleG2p g(dict);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = python_ipa_first_lines(wiki, static_cast<int>(src.size()));
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
