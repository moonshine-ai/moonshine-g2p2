#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p/utf8-utils.h"

using namespace moonshine_g2p;

TEST_CASE("utf8_split_codepoints ascii") {
  const auto v = utf8_split_codepoints("abc");
  REQUIRE(v.size() == 3);
  CHECK(v[0] == "a");
  CHECK(v[1] == "b");
  CHECK(v[2] == "c");
}

TEST_CASE("utf8_split_codepoints two-byte") {
  const std::string s = "\xC3\xA9";  // é
  const auto v = utf8_split_codepoints(s);
  REQUIRE(v.size() == 1);
  CHECK(v[0] == s);
}

TEST_CASE("utf8_find_token_codepoints") {
  const std::string text = "I read";
  const auto hit = utf8_find_token_codepoints(text, "read", 0);
  REQUIRE(hit);
  CHECK(hit->first == 2);
  CHECK(hit->second == 6);
}
