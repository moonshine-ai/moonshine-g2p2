#pragma once

#include <string>
#include <vector>

namespace moonshine_g2p {

// Python str.split() with no args: split on arbitrary whitespace runs.
std::vector<std::string> split_text_to_words(std::string_view text);

// Python normalize_word_for_lookup: lower, strip, trim non-alphanumeric from edges.
// Non-ASCII code points are treated as word characters (not trimmed).
std::string normalize_word_for_lookup(std::string_view token);

// CMUdict key: lowercase, strip trailing (digits) alternate suffix.
std::string normalize_grapheme_key(std::string_view word_token);

}  // namespace moonshine_g2p
