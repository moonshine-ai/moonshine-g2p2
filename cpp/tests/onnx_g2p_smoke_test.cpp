#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine_g2p/moonshine_g2p.hpp"
#include "moonshine_g2p/moonshine_onnx_g2p.hpp"

#include <cstdlib>
#include <filesystem>

using namespace moonshine_g2p;

TEST_CASE("MoonshineG2P ONNX fallback when MOONSHINE_G2P_MODELS_ROOT is set") {
  const char* root = std::getenv("MOONSHINE_G2P_MODELS_ROOT");
  if (root == nullptr || root[0] == '\0') {
    MESSAGE("skip: export MOONSHINE_G2P_MODELS_ROOT to models/en_us parent");
    return;
  }
  MoonshineG2POptions opt;
  opt.model_root = std::filesystem::path(root).parent_path();
  const std::string dialect_dir = std::filesystem::path(root).filename().string();
  MoonshineG2P g2p(dialect_dir, opt);
  CHECK(g2p.uses_onnx());
  CHECK_FALSE(g2p.uses_spanish_rules());
  CHECK_FALSE(g2p.uses_german_rules());
  const std::string out = g2p.text_to_ipa("Hello world!");
  CHECK(out.size() > 0);
}

TEST_CASE("onnx end-to-end when MOONSHINE_G2P_MODELS_ROOT is set") {
  const char* root = std::getenv("MOONSHINE_G2P_MODELS_ROOT");
  if (root == nullptr || root[0] == '\0') {
    MESSAGE("skip: export MOONSHINE_G2P_MODELS_ROOT to models/en_us parent");
    return;
  }
  const std::filesystem::path base(root);
  const auto dict_path = base / "dict_filtered_heteronyms.tsv";
  const auto het = base / "heteronym" / "model.onnx";
  const auto oov = base / "oov" / "model.onnx";
  if (!std::filesystem::exists(dict_path) || !std::filesystem::exists(het) ||
      !std::filesystem::exists(oov)) {
    MESSAGE("skip: missing dict or ONNX bundles under MOONSHINE_G2P_MODELS_ROOT");
    return;
  }

  MoonshineOnnxG2p g2p(dict_path, het, oov, false);
  const std::string out = g2p.text_to_ipa("Hello world!");
  CHECK(out.size() > 0);
}
