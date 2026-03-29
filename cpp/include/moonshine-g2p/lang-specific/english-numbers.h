#ifndef MOONSHINE_G2P_LANG_SPECIFIC_ENGLISH_NUMBERS_H
#define MOONSHINE_G2P_LANG_SPECIFIC_ENGLISH_NUMBERS_H

#include <optional>
#include <string>
#include <string_view>

namespace moonshine_g2p {

/// If *token* is a supported plain numeral, return IPA; else ``nullopt`` (see ``english_number_token_ipa``).
std::optional<std::string> english_number_token_ipa(std::string_view token);

}  // namespace moonshine_g2p

#endif  // MOONSHINE_G2P_LANG_SPECIFIC_ENGLISH_NUMBERS_H
