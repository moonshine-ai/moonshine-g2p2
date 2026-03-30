#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/lang-specific/english.h"
#include "rule-g2p-test-support.h"

#include <filesystem>
#include <string>
#include <vector>

namespace r = moonshine_g2p::rule_g2p_test;

namespace {

std::filesystem::path resolve_en_dict(const std::filesystem::path& repo) {
  std::filesystem::path d = repo / "models" / "en_us" / "dict_filtered_heteronyms.tsv";
  if (std::filesystem::is_regular_file(d)) {
    return d;
  }
  return repo / "data" / "en_us" / "dict_filtered_heteronyms.tsv";
}

std::filesystem::path en_homograph_json(const std::filesystem::path& repo) {
  return repo / "models" / "en_us" / "heteronym" / "homograph_index.json";
}

std::vector<std::string> python_ipa_first_lines(const std::filesystem::path& text_file, int n) {
  return r::python_ref_first_lines(r::repo_root_from_tests_cpp(__FILE__), "english_g2p_ref.py", text_file,
                                   n);
}

bool python_english_import_ok() {
  return r::python_import_ok(r::repo_root_from_tests_cpp(__FILE__),
                             "from english_rule_g2p import EnglishLexiconRuleG2p");
}

}  // namespace

TEST_CASE("english: dialect_resolves_to_english_rules") {
  using moonshine_g2p::dialect_resolves_to_english_rules;
  CHECK(dialect_resolves_to_english_rules("en_us"));
  CHECK(dialect_resolves_to_english_rules("en-US"));
  CHECK(dialect_resolves_to_english_rules("EN_US"));
  CHECK(dialect_resolves_to_english_rules("english"));
  CHECK_FALSE(dialect_resolves_to_english_rules("de"));
}

TEST_CASE("english: wiki-text first 100 lines match Python when data and python3 exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = resolve_en_dict(repo);
  const std::filesystem::path wiki = repo / "data" / "en_us" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !python_english_import_ok()) {
    return;
  }
  const std::filesystem::path homograph = en_homograph_json(repo);
  moonshine_g2p::EnglishRuleG2p g(dict, homograph, std::nullopt, std::nullopt);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = python_ipa_first_lines(wiki, static_cast<int>(src.size()));
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
