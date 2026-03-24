#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine_g2p/ipa_postprocess.hpp"

using namespace moonshine_g2p;

TEST_CASE("levenshtein_distance") {
  CHECK(levenshtein_distance({}, {}) == 0);
  CHECK(levenshtein_distance({"a"}, {}) == 1);
  CHECK(levenshtein_distance({"a"}, {"a"}) == 0);
  CHECK(levenshtein_distance({"a", "b"}, {"a", "c"}) == 1);
}

TEST_CASE("pick_closest_cmudict_ipa single") {
  const std::vector<std::string> pred = {"x"};
  CHECK(pick_closest_cmudict_ipa(pred, {"only"}, 0) == "only");
}

TEST_CASE("match_prediction_to_cmudict_ipa") {
  const std::vector<std::string> alts = {"həˈloʊ", "həˈləʊ"};
  const auto m = match_prediction_to_cmudict_ipa("həˈloʊ", alts);
  REQUIRE(m);
  CHECK(*m == "həˈloʊ");
}
