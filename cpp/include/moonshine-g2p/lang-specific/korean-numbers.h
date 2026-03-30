#ifndef MOONSHINE_G2P_LANG_SPECIFIC_KOREAN_NUMBERS_H
#define MOONSHINE_G2P_LANG_SPECIFIC_KOREAN_NUMBERS_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace moonshine_g2p {

/// Non-negative integer → Sino-Korean cardinal Hangul (``영`` … ``경`` …), mirroring
/// ``korean_numbers.int_to_sino_korean_hangul``.
std::string int_to_sino_korean_hangul(std::uint64_t n);

/// If *token* is a plain ASCII numeral (optional sign; ``.``/``,`` decimal; strip ``,`` / ``_``),
/// return G2P fragments (Hangul and optionally ``마이너스``). Otherwise ``std::nullopt``.
std::optional<std::vector<std::string>> korean_reading_fragments_from_ascii_numeral_token(
    std::string_view token);

bool is_ascii_numeral_token(std::string_view token);

}  // namespace moonshine_g2p

#endif  // MOONSHINE_G2P_LANG_SPECIFIC_KOREAN_NUMBERS_H
