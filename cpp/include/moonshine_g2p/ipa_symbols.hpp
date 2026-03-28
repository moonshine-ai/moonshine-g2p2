#pragma once

#include <string>

namespace moonshine_g2p::ipa {

/// UTF-8 encoding of U+02C8 (combining-like primary stress for IPA output).
inline const std::string kPrimaryStressUtf8{"\xCB\x88"};
/// UTF-8 encoding of U+02CC (secondary stress).
inline const std::string kSecondaryStressUtf8{"\xCB\x8C"};

}  // namespace moonshine_g2p::ipa
