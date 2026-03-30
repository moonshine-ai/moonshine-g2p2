#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/japanese-onnx-g2p.h"
#include "rule-g2p-test-support.h"

namespace r = moonshine_g2p::rule_g2p_test;

TEST_CASE("japanese onnx g2p: first 100 wiki IPA lines match Python when assets exist") {
  constexpr std::size_t kWikiLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto model = moonshine_g2p::default_japanese_tok_pos_model_dir(repo);
  const auto dict = moonshine_g2p::default_japanese_dict_path(repo);
  const auto wiki = repo / "data" / "ja" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(model / "model.onnx") ||
      !std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki)) {
    return;
  }
  moonshine_g2p::JapaneseOnnxG2p g2p(model, dict, false);
  const auto src = r::read_text_first_lines(wiki, kWikiLines);
  const std::vector<std::string> py =
      r::python_ref_first_lines(repo, "japanese_onnx_g2p_ref.py", wiki, static_cast<int>(src.size()));
  REQUIRE(py.size() == src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g2p.text_to_ipa(src[i]) == py[i]);
  }
}
