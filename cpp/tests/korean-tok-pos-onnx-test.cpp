#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/korean-tok-pos-onnx.h"
#include "rule-g2p-test-support.h"

#include <chrono>
#include <fstream>

namespace r = moonshine_g2p::rule_g2p_test;

TEST_CASE("korean tok pos: single sentence matches Python reference") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto model = moonshine_g2p::default_korean_tok_pos_model_dir(repo);
  if (!std::filesystem::is_regular_file(model / "model.onnx")) {
    return;
  }
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto tmp = std::filesystem::temp_directory_path() /
                   ("moonshine_ko_tok_one_" + std::to_string(tick) + ".txt");
  {
    std::ofstream o(tmp);
    o << "대한민국의 수도는 서울이다.\n";
  }
  const std::vector<std::string> py =
      r::python_ref_first_lines(repo, "korean_tok_pos_ref.py", tmp, 1);
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  REQUIRE(py.size() == 1);
  moonshine_g2p::KoreanTokPosOnnx pipe(model, false);
  const auto pairs = pipe.annotate("대한민국의 수도는 서울이다.");
  CHECK(moonshine_g2p::KoreanTokPosOnnx::format_annotated_line(pairs) == py[0]);
}

TEST_CASE("korean tok pos: first 100 wiki lines match Python when assets exist") {
  constexpr std::size_t kWikiLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto model = moonshine_g2p::default_korean_tok_pos_model_dir(repo);
  const auto wiki = repo / "data" / "ko" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(model / "model.onnx") ||
      !std::filesystem::is_regular_file(wiki)) {
    return;
  }
  moonshine_g2p::KoreanTokPosOnnx pipe(model, false);
  const auto src = r::read_text_first_lines(wiki, kWikiLines);
  const std::vector<std::string> py =
      r::python_ref_first_lines(repo, "korean_tok_pos_ref.py", wiki, static_cast<int>(src.size()));
  REQUIRE(py.size() == src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    const auto pairs = pipe.annotate(src[i]);
    CHECK(moonshine_g2p::KoreanTokPosOnnx::format_annotated_line(pairs) == py[i]);
  }
}
