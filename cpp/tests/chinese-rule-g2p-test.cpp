#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/lang-specific/chinese.h"
#ifdef MOONSHINE_G2P_WITH_MOONSHINE_G2P_CLASS
#include "moonshine-g2p/moonshine-g2p.h"
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path make_temp_tsv(const char* contents) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path p = std::filesystem::temp_directory_path() /
                            ("moonshine_chinese_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

}  // namespace

TEST_CASE("chinese: dialect_resolves_to_chinese_rules") {
  using moonshine_g2p::dialect_resolves_to_chinese_rules;
  CHECK(dialect_resolves_to_chinese_rules("zh"));
  CHECK(dialect_resolves_to_chinese_rules("zh-Hans"));
  CHECK(dialect_resolves_to_chinese_rules("ZH_cn"));
  CHECK(dialect_resolves_to_chinese_rules("cmn"));
  CHECK_FALSE(dialect_resolves_to_chinese_rules("ja"));
}

TEST_CASE("chinese: lexicon lookup and Arabic numeral expansion via per-char Han IPA") {
  // 42 → 四十二; provide per-character entries for han_reading_to_ipa.
  const auto p = make_temp_tsv(
      "\xe5\x9b\x9b\ts4_ipa\n"
      "\xe5\x8d\x81\tsh2_ipa\n"
      "\xe4\xba\x8c\ter2_ipa\n"
      "\xe6\xb5\x8b\xe8\xaf\x95\toneword_ipa\n");
  moonshine_g2p::ChineseRuleG2p g(p);
  CHECK(g.word_to_ipa("\xe6\xb5\x8b\xe8\xaf\x95") == "oneword_ipa");
  CHECK(g.word_to_ipa("42") == "s4_ipa sh2_ipa er2_ipa");
  std::filesystem::remove(p);
}

#ifdef MOONSHINE_G2P_WITH_MOONSHINE_G2P_CLASS
TEST_CASE("chinese: MoonshineG2P zh uses rule backend and matches ChineseRuleG2p") {
  const auto p = make_temp_tsv(
      "\xe6\xb5\x8b\xe8\xaf\x95\tzh_test_ipa\n");
  moonshine_g2p::MoonshineG2POptions opt;
  opt.chinese_dict_path = p;
  moonshine_g2p::MoonshineG2P g("zh", opt);
  CHECK(g.uses_chinese_rules());
  CHECK_FALSE(g.uses_onnx());
  CHECK(g.dialect_id() == "zh-Hans");
  moonshine_g2p::ChineseRuleG2p direct(p);
  const std::string w = "\xe6\xb5\x8b\xe8\xaf\x95";
  CHECK(g.text_to_ipa(w) == direct.text_to_ipa(w));
  std::filesystem::remove(p);
}
#endif
