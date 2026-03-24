#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace moonshine_g2p {

void utf8_append_codepoint(std::string& out, char32_t cp);

std::vector<std::string> utf8_split_codepoints(const std::string& utf8);

// Find *token* as a contiguous subsequence of code points in *text*, starting at
// code point index >= start_cp. Returns [start, end) code point indices into *text*.
std::optional<std::pair<int, int>> utf8_find_token_codepoints(const std::string& text,
                                                              const std::string& token,
                                                              int start_cp);

}  // namespace moonshine_g2p
