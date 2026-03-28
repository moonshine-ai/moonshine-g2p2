#ifndef MOONSHINE_G2P_G2P_WORD_LOG_H
#define MOONSHINE_G2P_G2P_WORD_LOG_H

#include <string>
#include <vector>

namespace moonshine_g2p {

// How a surface word was converted to IPA in MoonshineOnnxG2p::text_to_ipa.
enum class G2pWordPath {
  kSkippedEmptyKey,  // normalize_word_for_lookup produced nothing
  kTokenNotLocatedInText,  // could not align token in full text (skipped)
  kRuleBasedG2p,  // language-specific rule engine (e.g. Spanish grapheme rules)
  kDictUnambiguous,  // exactly one CMUdict pronunciation
  kDictHeteronym,  // multiple pronunciations; heteronym ONNX chose one
  kDictFirstAlternativeNoHeteronymModel,  // multiple pronunciations; no model — first alt
  kOovModel,  // OOV ONNX produced phoneme tokens
  kOovModelNoOutput,  // OOV ran but produced no non-empty IPA
  kUnknownNoOovModel,  // not in dict and no OOV model loaded
};

const char* g2p_word_path_tag(G2pWordPath path);

struct G2pWordLog {
  std::string surface_token;
  std::string lookup_key;
  G2pWordPath path = G2pWordPath::kSkippedEmptyKey;
  /// Resulting IPA fragment for this word (may be empty).
  std::string ipa;
};

/// One-line human-readable summary (for stderr logging).
std::string format_g2p_word_log_line(const G2pWordLog& entry);

}  // namespace moonshine_g2p

#endif  // MOONSHINE_G2P_G2P_WORD_LOG_H
