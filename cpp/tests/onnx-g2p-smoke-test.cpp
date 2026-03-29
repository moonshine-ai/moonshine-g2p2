#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/moonshine-g2p.h"

#include <cstdlib>
#include <filesystem>

using namespace moonshine_g2p;

TEST_CASE("MoonshineG2P en_us rule-based when MOONSHINE_G2P_MODELS_ROOT is set") {
  const char* root = std::getenv("MOONSHINE_G2P_MODELS_ROOT");
  if (root == nullptr || root[0] == '\0') {
    MESSAGE("skip: export MOONSHINE_G2P_MODELS_ROOT to models/en_us parent");
    return;
  }
  MoonshineG2POptions opt;
  opt.model_root = std::filesystem::path(root).parent_path();
  const std::string dialect_dir = std::filesystem::path(root).filename().string();
  MoonshineG2P g2p(dialect_dir, opt);
  CHECK_FALSE(g2p.uses_onnx());
  CHECK(g2p.uses_english_rules());
  CHECK_FALSE(g2p.uses_spanish_rules());
  CHECK_FALSE(g2p.uses_german_rules());
  const std::string out = g2p.text_to_ipa("Hello world!");
  CHECK(out.size() > 0);
}
