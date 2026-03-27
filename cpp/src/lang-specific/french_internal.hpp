#pragma once

#include <string>

namespace moonshine_g2p::french_detail {

/// Rule-based OOV IPA (mirrors ``french_oov_rules.oov_word_to_ipa``).
[[nodiscard]] std::string oov_word_to_ipa(const std::string& word, bool with_stress);

}  // namespace moonshine_g2p::french_detail
