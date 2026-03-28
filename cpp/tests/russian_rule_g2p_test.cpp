#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine_g2p/lang-specific/russian.hpp"
#ifdef MOONSHINE_G2P_WITH_MOONSHINE_G2P_CLASS
#include "moonshine_g2p/moonshine_g2p.hpp"
#endif

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
      std::filesystem::temp_directory_path() / ("moonshine_russian_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

std::filesystem::path repo_root_from_this_file() {
  return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
}

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

std::string python_ipa_from_file(const std::filesystem::path& utf8_file) {
  const std::filesystem::path repo = repo_root_from_this_file();
  const std::filesystem::path script = repo / "cpp" / "tests" / "russian_g2p_ref.py";
  std::ostringstream cmd;
  cmd << "env PYTHONPATH=" << std::quoted(repo.string()) << " python3 " << std::quoted(script.string()) << " "
      << std::quoted(utf8_file.string());
  return shell_capture(cmd.str());
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
  const std::filesystem::path repo = repo_root_from_this_file();
  const std::filesystem::path script = repo / "cpp" / "tests" / "russian_g2p_ref.py";
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

bool python_russian_import_ok() {
  const std::filesystem::path repo = repo_root_from_this_file();
  std::ostringstream cmd;
  cmd << "env PYTHONPATH=" << std::quoted(repo.string())
      << " python3 -c \"from russian_rule_g2p import text_to_ipa\"";
  return system(cmd.str().c_str()) == 0;
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
  const std::filesystem::path dict = repo_root_from_this_file() / "data" / "ru" / "dict.tsv";
  if (!std::filesystem::is_regular_file(dict) || !python_russian_import_ok()) {
    return;
  }
  moonshine_g2p::RussianRuleG2p g(dict);
  const std::string py = python_ipa_one_line(
      "\xD0\x9B\xD0\xB8\xD1\x82\xD0\xB2\xD0\xB0");
  CHECK(g.word_to_ipa(
            "\xD0\x9B\xD0\xB8\xD1\x82\xD0\xB2\xD0\xB0") == py);
}

TEST_CASE("russian: wiki-text first 100 lines match Python when data and python3 exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const std::filesystem::path dict = repo_root_from_this_file() / "data" / "ru" / "dict.tsv";
  const std::filesystem::path wiki = repo_root_from_this_file() / "data" / "ru" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !python_russian_import_ok()) {
    return;
  }
  moonshine_g2p::RussianRuleG2p g(dict);
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
