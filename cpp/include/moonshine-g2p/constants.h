#ifndef MOONSHINE_G2P_CONSTANTS_H
#define MOONSHINE_G2P_CONSTANTS_H

#include <string_view>

namespace moonshine_g2p {

inline constexpr std::string_view kSpecialPad = "<pad>";
inline constexpr std::string_view kSpecialUnk = "<unk>";
inline constexpr std::string_view kPhonPad = "<pad>";
inline constexpr std::string_view kPhonBos = "<bos>";
inline constexpr std::string_view kPhonEos = "<eos>";

inline constexpr int kHeteronymContextMaxChars = 32;
inline constexpr int kConfigOnnxSchemaVersion = 1;
inline constexpr std::string_view kConfigOnnxFileName = "onnx-config.json";

}  // namespace moonshine_g2p

#endif  // MOONSHINE_G2P_CONSTANTS_H
