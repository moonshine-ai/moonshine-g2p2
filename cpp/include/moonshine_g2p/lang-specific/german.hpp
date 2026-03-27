#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonshine_g2p {

struct G2pWordLog;

/// Rule- and lexicon-based German G2P (High German), mirroring ``german_rule_g2p.py``.
class GermanRuleG2p {
 public:
  struct Options {
    bool with_stress = true;
    /// When true (default), move ˈ/ˌ before the syllable nucleus (vocoder-style).
    bool vocoder_stress = true;
  };

  /// Load ``word<TAB>IPA`` TSV; throws ``std::runtime_error`` if the file is missing/unreadable.
  explicit GermanRuleG2p(std::filesystem::path dict_tsv);
  explicit GermanRuleG2p(std::filesystem::path dict_tsv, Options options);

  [[nodiscard]] const std::string& dialect_id() const { return dialect_id_; }

  /// Single surface word (no spaces); punctuation should be stripped by the caller.
  [[nodiscard]] std::string word_to_ipa(const std::string& word) const;

  /// Tokenize like Python ``text_to_ipa`` (German letters, digits, hyphen; preserve other chars).
  [[nodiscard]] std::string text_to_ipa(const std::string& text,
                                         std::vector<G2pWordLog>* per_word_log = nullptr) const;

  /// Exported for tests / tooling (same as the Python helper).
  [[nodiscard]] static std::string normalize_ipa_stress_for_vocoder(std::string ipa);

 private:
  std::string dialect_id_{"de-DE"};
  Options options_;
  std::unordered_map<std::string, std::string> lexicon_;

  [[nodiscard]] std::string lookup_or_rules(const std::string& raw_word) const;
  [[nodiscard]] std::string finalize_ipa(std::string ipa) const;
};

/// True for ``de``, ``de-DE``, ``de_de``, ``german`` (case-insensitive).
[[nodiscard]] bool dialect_resolves_to_german_rules(std::string_view dialect_id);

}  // namespace moonshine_g2p
