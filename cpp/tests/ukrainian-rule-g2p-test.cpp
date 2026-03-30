#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/lang-specific/ukrainian.h"
#include "rule-g2p-test-support.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace r = moonshine_g2p::rule_g2p_test;

namespace {

bool python_uk_import_ok() {
  return r::python_import_ok(r::repo_root_from_tests_cpp(__FILE__), "from ukrainian_rule_g2p import text_to_ipa");
}

void check_wiki_parity(const std::filesystem::path& wiki) {
  constexpr std::size_t kWikiParityLines = 100;
  if (!std::filesystem::is_regular_file(wiki) || !python_uk_import_ok()) {
    return;
  }
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py =
      r::python_ref_first_lines(r::repo_root_from_tests_cpp(__FILE__), "ukrainian_g2p_ref.py", wiki,
                                static_cast<int>(src.size()), {}, {});
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(moonshine_g2p::ukrainian_text_to_ipa(src[i]) == py[i]);
  }
}

}  // namespace

TEST_CASE("ukrainian: m'ясо and кінь match Python when python3 exists") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  if (!python_uk_import_ok()) {
    return;
  }
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path tmp =
      std::filesystem::temp_directory_path() / ("uk_g2p_" + std::to_string(tick) + ".txt");
  {
    std::ofstream o(tmp, std::ios::binary);
    o << "м'ясо кінь";
  }
  const std::vector<std::string> py =
      r::python_ref_first_lines(repo, "ukrainian_g2p_ref.py", tmp, 1, {}, {});
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  REQUIRE(py.size() == 1);
  CHECK(moonshine_g2p::ukrainian_text_to_ipa("м'ясо кінь") == py[0]);
}

TEST_CASE("ukrainian: dialect ids include uk and uk-UA") {
  const auto ids = moonshine_g2p::UkrainianRuleG2p::dialect_ids();
  CHECK(std::find(ids.begin(), ids.end(), "uk") != ids.end());
  CHECK(std::find(ids.begin(), ids.end(), "uk-UA") != ids.end());
}

TEST_CASE("ukrainian: wiki-text first 100 lines match Python when data and python3 exist") {
  const std::filesystem::path wiki = r::repo_root_from_tests_cpp(__FILE__) / "data" / "uk" / "wiki-text.txt";
  check_wiki_parity(wiki);
}
