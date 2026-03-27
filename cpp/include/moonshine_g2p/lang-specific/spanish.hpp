#pragma once

#include <string>
#include <vector>

namespace moonshine_g2p {

/// Pronunciation preset for rule-based Spanish G2P (mirrors ``spanish_rule_g2p.SpanishDialect``).
struct SpanishDialect {
  std::string id;
  std::string ce_ci_z_ipa; // "s" (seseo) or "θ" (distinción)
  bool yeismo = true;
  std::string y_consonant_ipa; // e.g. "ʝ" or "ʒ"
  std::string ll_ipa;          // e.g. "ʎ" when yeismo is false
  std::string x_intervocalic_default = "ks";
  std::string x_initial_before_vowel = "s";
  std::string voiceless_velar_fricative; // "x" or "h"
  std::string trill_ipa = "r";
  std::string tap_ipa; // e.g. "ɾ"
  bool nasal_assimilation = false;
  bool narrow_intervocalic_obstruents = true;
  enum class CodaS { Keep, H, Drop } coda_s_mode = CodaS::Keep;
};

/// Sorted CLI ids (same set as Python ``dialect_ids()``).
std::vector<std::string> spanish_dialect_cli_ids();

/// Build dialect from CLI id, or throw ``std::invalid_argument``.
SpanishDialect spanish_dialect_from_cli_id(const std::string &cli_id,
                                           bool narrow_intervocalic_obstruents = true);

/// One word → IPA (letters outside Spanish set stripped; punctuation should be handled by caller).
std::string spanish_word_to_ipa(const std::string &word, const SpanishDialect &dialect,
                                bool with_stress = true);

/// Tokenize like Python ``text_to_ipa`` (Spanish letters + digits; collapse spaces).
std::string spanish_text_to_ipa(const std::string &text, const SpanishDialect &dialect,
                                  bool with_stress = true);

} // namespace moonshine_g2p
