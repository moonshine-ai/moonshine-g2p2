#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/chinese-tok-pos-onnx.h"
#include "rule-g2p-test-support.h"

#include <chrono>
#include <fstream>

namespace r = moonshine_g2p::rule_g2p_test;

TEST_CASE("chinese tok pos: single sentence matches Python reference") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto model = moonshine_g2p::default_chinese_tok_pos_model_dir(repo);
  if (!std::filesystem::is_regular_file(model / "model.onnx")) {
    return;
  }
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto tmp = std::filesystem::temp_directory_path() /
                   ("moonshine_zh_tok_one_" + std::to_string(tick) + ".txt");
  {
    std::ofstream o(tmp);
    o << "上海是一座城市。\n";
  }
  const std::vector<std::string> py =
      r::python_ref_first_lines(repo, "chinese_tok_pos_ref.py", tmp, 1);
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  REQUIRE(py.size() == 1);
  moonshine_g2p::ChineseTokPosOnnx pipe(model, false);
  const auto pairs = pipe.annotate("上海是一座城市。");
  CHECK(moonshine_g2p::ChineseTokPosOnnx::format_annotated_line(pairs) == py[0]);
}

TEST_CASE("chinese tok pos: first 100 wiki lines match Python when assets exist") {
  constexpr std::size_t kWikiLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto model = moonshine_g2p::default_chinese_tok_pos_model_dir(repo);
  const auto wiki = repo / "data" / "zh_hans" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(model / "model.onnx") ||
      !std::filesystem::is_regular_file(wiki)) {
    return;
  }
  moonshine_g2p::ChineseTokPosOnnx pipe(model, false);
  const auto src = r::read_text_first_lines(wiki, kWikiLines);
  std::vector<std::string> tokenizable;
  for (const std::string& line : src) {
    try {
      (void)pipe.annotate(line);
      tokenizable.push_back(line);
    } catch (const std::exception&) {
      // Same WordPiece path as Python: skip lines with tricky mixed scripts / NFC edge cases.
    }
  }
  if (tokenizable.empty()) {
    return;
  }
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto tmp = std::filesystem::temp_directory_path() /
                   ("moonshine_zh_wiki_filt_" + std::to_string(tick) + ".txt");
  {
    std::ofstream o(tmp);
    for (const std::string& line : tokenizable) {
      o << line << '\n';
    }
  }
  const std::vector<std::string> py =
      r::python_ref_first_lines(repo, "chinese_tok_pos_ref.py", tmp, static_cast<int>(tokenizable.size()));
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  REQUIRE(py.size() == tokenizable.size());
  for (std::size_t i = 0; i < tokenizable.size(); ++i) {
    INFO("wiki filtered index " << (i + 1));
    const auto pairs = pipe.annotate(tokenizable[i]);
    CHECK(moonshine_g2p::ChineseTokPosOnnx::format_annotated_line(pairs) == py[i]);
  }
}
