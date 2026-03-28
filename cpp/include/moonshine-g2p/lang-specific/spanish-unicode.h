#ifndef MOONSHINE_G2P_LANG_SPECIFIC_SPANISH_UNICODE_H
#define MOONSHINE_G2P_LANG_SPECIFIC_SPANISH_UNICODE_H

#include <cstdint>
#include <string>

namespace moonshine_g2p::spanish_unicode {

[[nodiscard]] std::string utf32_to_utf8(const std::u32string& u);
[[nodiscard]] std::u32string utf8_to_utf32(const std::string& s);
[[nodiscard]] std::string unicode_lower_utf8(const std::string& s);
[[nodiscard]] std::string strip_accents_utf8(const std::string& s);
[[nodiscard]] std::string word_key(const std::string& wraw);
[[nodiscard]] bool is_word_char(char32_t cp);
[[nodiscard]] bool is_space_char(char32_t cp);
/// UTF-8 replacement from the strip table, or nullptr when absent.
[[nodiscard]] const char* strip_replacement_utf8(char32_t cp);

}  // namespace moonshine_g2p::spanish_unicode

#endif  // MOONSHINE_G2P_LANG_SPECIFIC_SPANISH_UNICODE_H
