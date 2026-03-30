#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/lang-specific/arabic.h"
#include "rule-g2p-test-support.h"

namespace r = moonshine_g2p::rule_g2p_test;

TEST_CASE("arabic rule g2p: first 100 wiki lines match Python when assets exist") {
  constexpr std::size_t kWikiLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto model = moonshine_g2p::resolve_arabic_onnx_model_dir(repo / "models");
  const auto dict = moonshine_g2p::resolve_arabic_dict_path(repo / "models");
  const auto wiki = repo / "data" / "ar" / "wiki-text.txt";
  if (!std::filesystem::is_regular_file(model / "model.onnx") ||
      !std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki)) {
    return;
  }
  moonshine_g2p::ArabicRuleG2p g2p(model, dict, false);
  const auto src = r::read_text_first_lines(wiki, kWikiLines);
  const std::vector<std::string> py =
      r::python_ref_first_lines(repo, "arabic_rule_g2p_ref.py", wiki, static_cast<int>(src.size()));
  REQUIRE(py.size() == src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g2p.text_to_ipa(src[i]) == py[i]);
  }
}
