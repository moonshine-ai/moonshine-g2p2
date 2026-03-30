#ifndef MOONSHINE_G2P_JAPANESE_KANA_TO_IPA_H
#define MOONSHINE_G2P_JAPANESE_KANA_TO_IPA_H

#include <string>
#include <string_view>

namespace moonshine_g2p {

/// NFKC katakana/hiragana (and prolonged mark) → IPA, matching ``japanese_kana_to_ipa.py``.
std::string katakana_hiragana_to_ipa(std::string_view utf8);

bool japanese_is_kana_only(std::string_view utf8);
bool japanese_has_japanese_script(std::string_view utf8);

}  // namespace moonshine_g2p

#endif
