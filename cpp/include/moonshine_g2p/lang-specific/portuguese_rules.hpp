#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace moonshine_g2p::portuguese_rules {

[[nodiscard]] char32_t pt_tolower(char32_t c);
[[nodiscard]] std::string normalize_lookup_key_utf8(const std::string& word);

[[nodiscard]] std::optional<std::string> roman_numeral_token_to_ipa(const std::string& letters_lower,
                                                                   bool is_pt_pt);
[[nodiscard]] std::string rules_word_to_ipa_utf8(const std::string& raw, bool is_pt_pt, bool with_stress);
[[nodiscard]] std::string pt_pt_apply_rules_final_s_to_esh(std::string ipa, const std::string& letters_key);

[[nodiscard]] const std::unordered_map<std::string, std::string>& fw_pt();
[[nodiscard]] const std::unordered_map<std::string, std::string>& fw_br();
[[nodiscard]] const std::unordered_map<std::string, std::string>& sc_straddle();

}  // namespace moonshine_g2p::portuguese_rules
