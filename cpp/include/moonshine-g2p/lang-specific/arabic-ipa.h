#ifndef MOONSHINE_G2P_LANG_SPECIFIC_ARABIC_IPA_H
#define MOONSHINE_G2P_LANG_SPECIFIC_ARABIC_IPA_H

#include <string>
#include <string_view>

namespace moonshine_g2p {

std::string arabic_msa_strip_diacritics_utf8(std::string_view utf8);
std::string arabic_msa_apply_onnx_partial_postprocess_utf8(std::string_view utf8);
std::string arabic_msa_word_to_ipa_with_assimilation_utf8(std::string_view utf8);

}  // namespace moonshine_g2p

#endif
