#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/lang-specific/spanish.h"
#include "rule-g2p-test-support.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace r = moonshine_g2p::rule_g2p_test;

namespace {

std::vector<std::string> python_ipa_first_lines(const std::filesystem::path& text_file, int n,
                                                const char* dialect_cli) {
  std::vector<std::string> before;
  if (dialect_cli != nullptr && dialect_cli[0] != '\0') {
    before = {"--dialect", dialect_cli};
  }
  return r::python_ref_first_lines(r::repo_root_from_tests_cpp(__FILE__), "spanish_g2p_ref.py", text_file, n,
                                   before);
}

bool python_spanish_import_ok() {
  return r::python_import_ok(r::repo_root_from_tests_cpp(__FILE__), "from spanish_rule_g2p import text_to_ipa");
}

void check_wiki_parity(const std::filesystem::path& wiki, const char* dialect_cli,
                       const moonshine_g2p::SpanishDialect& dialect) {
  constexpr std::size_t kWikiParityLines = 100;
  if (!std::filesystem::is_regular_file(wiki) || !python_spanish_import_ok()) {
    return;
  }
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py =
      python_ipa_first_lines(wiki, static_cast<int>(src.size()), dialect_cli);
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(moonshine_g2p::spanish_text_to_ipa(src[i], dialect) == py[i]);
  }
}

}  // namespace

TEST_CASE("spanish: En 1891 matches Python when python3 exists") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  if (!python_spanish_import_ok()) {
    return;
  }
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path tmp =
      std::filesystem::temp_directory_path() / ("es_g2p_digits_" + std::to_string(tick) + ".txt");
  {
    std::ofstream o(tmp, std::ios::binary);
    o << "En 1891";
  }
  const std::vector<std::string> py =
      r::python_ref_first_lines(repo, "spanish_g2p_ref.py", tmp, 1, {"--dialect", "es-MX"});
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  REQUIRE(py.size() == 1);
  const auto dialect = moonshine_g2p::spanish_dialect_from_cli_id("es-MX");
  CHECK(moonshine_g2p::spanish_text_to_ipa("En 1891", dialect) == py[0]);
}

TEST_CASE("spanish: dialect ids include es-MX and es-ES") {
  const auto ids = moonshine_g2p::spanish_dialect_cli_ids();
  CHECK(std::find(ids.begin(), ids.end(), "es-MX") != ids.end());
  CHECK(std::find(ids.begin(), ids.end(), "es-ES") != ids.end());
}

TEST_CASE("spanish: wiki-text first 100 lines es_mx match Python when data and python3 exist") {
  const std::filesystem::path wiki = r::repo_root_from_tests_cpp(__FILE__) / "data" / "es_mx" / "wiki-text.txt";
  const auto dialect = moonshine_g2p::spanish_dialect_from_cli_id("es-MX");
  check_wiki_parity(wiki, nullptr, dialect);
}

TEST_CASE("spanish: wiki-text first 100 lines es_es match Python when data and python3 exist") {
  const std::filesystem::path wiki = r::repo_root_from_tests_cpp(__FILE__) / "data" / "es_es" / "wiki-text.txt";
  const auto dialect = moonshine_g2p::spanish_dialect_from_cli_id("es-ES");
  check_wiki_parity(wiki, "es-ES", dialect);
}
