#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine_g2p/lang-specific/italian.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path make_temp_tsv(const char* contents) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path p =
      std::filesystem::temp_directory_path() / ("moonshine_italian_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

std::filesystem::path repo_root_from_this_file() {
  return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
}

/// Capture stdout from a shell command (trim trailing newline only on last line aggregation).
std::string shell_capture(const std::string& cmd) {
  FILE* pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    return {};
  }
  std::string out;
  char buf[8192];
  while (fgets(buf, sizeof(buf), pipe) != nullptr) {
    out += buf;
  }
  (void)pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  return out;
}

/// Run ``italian_g2p_ref.py`` on a UTF-8 file containing one logical line (no trailing newline required).
std::string python_ipa_from_file(const std::filesystem::path& utf8_file) {
  const std::filesystem::path repo = repo_root_from_this_file();
  const std::filesystem::path script = repo / "cpp" / "tests" / "italian_g2p_ref.py";
  std::ostringstream cmd;
  cmd << "env PYTHONPATH=" << std::quoted(repo.string()) << " python3 " << std::quoted(script.string()) << " "
      << std::quoted(utf8_file.string());
  return shell_capture(cmd.str());
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
  const std::filesystem::path repo = repo_root_from_this_file();
  const std::filesystem::path script = repo / "cpp" / "tests" / "italian_g2p_ref.py";
  std::ostringstream cmd;
  cmd << "env PYTHONPATH=" << std::quoted(repo.string()) << " python3 " << std::quoted(script.string()) << " "
      << std::quoted(text_file.string()) << " --first-lines " << n;
  const std::string block = shell_capture(cmd.str());
  std::vector<std::string> lines;
  std::istringstream iss(block);
  std::string L;
  while (std::getline(iss, L)) {
    if (!L.empty() && L.back() == '\r') {
      L.pop_back();
    }
    lines.push_back(std::move(L));
  }
  return lines;
}

bool python_italian_import_ok() {
  const std::filesystem::path repo = repo_root_from_this_file();
  std::ostringstream cmd;
  cmd << "env PYTHONPATH=" << std::quoted(repo.string())
      << " python3 -c \"from italian_rule_g2p import text_to_ipa\"";
  return system(cmd.str().c_str()) == 0;
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
  const std::filesystem::path dict = repo_root_from_this_file() / "data" / "it" / "dict.tsv";
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
  const std::filesystem::path dict = repo_root_from_this_file() / "data" / "it" / "dict.tsv";
  const std::filesystem::path wiki = repo_root_from_this_file() / "data" / "it" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !python_italian_import_ok()) {
    return;
  }
  moonshine_g2p::ItalianRuleG2p g(dict);
  std::ifstream in(wiki);
  REQUIRE(in);
  std::vector<std::string> src;
  std::string line;
  while (src.size() < kWikiParityLines && std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    src.push_back(std::move(line));
  }
  const std::vector<std::string> py = python_ipa_first_lines(wiki, static_cast<int>(src.size()));
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
