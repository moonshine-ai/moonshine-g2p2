#ifndef MOONSHINE_G2P_MOONSHINE_G2P_H
#define MOONSHINE_G2P_MOONSHINE_G2P_H

#include "moonshine-g2p/g2p-word-log.h"
#include "moonshine-g2p/moonshine-g2p-options.h"
#include "moonshine-g2p/rule-based-g2p-factory.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_g2p {

class RuleBasedG2p;

/// True if *dialect_id* maps to the built-in Spanish rule engine (e.g. ``es-MX``, ``es_ar``).
bool dialect_resolves_to_spanish_rules(
    std::string_view dialect_id, bool spanish_narrow_obstruents = true);

/// True if *dialect_id* is recognized as a rule-based dialect (Spanish narrow-obstruent flag from
/// *options*). Does not check lexicon paths; ``MoonshineG2P`` may still throw if files are missing.
bool dialect_uses_rule_based_g2p(std::string_view dialect_id,
                                               const MoonshineG2POptions& options = {});

/// Single entry point: *dialect_id* is a tag such as ``en_us``, ``es-AR``, ``de``, ``de-DE``.
/// Only rule-based engines are supported (English, Spanish, German, French, Dutch, Italian, Russian,
/// Chinese, Portuguese). Other dialect ids throw ``std::runtime_error``.
class MoonshineG2P {
 public:
  explicit MoonshineG2P(std::string dialect_id, MoonshineG2POptions options = {});
  ~MoonshineG2P();

  MoonshineG2P(const MoonshineG2P&) = delete;
  MoonshineG2P& operator=(const MoonshineG2P&) = delete;
  MoonshineG2P(MoonshineG2P&&) noexcept;
  MoonshineG2P& operator=(MoonshineG2P&&) noexcept;

  std::string text_to_ipa(std::string_view text,
                                        std::vector<G2pWordLog>* per_word_log = nullptr);

  bool uses_spanish_rules() const {
    return rule_backend_ == RuleBasedG2pKind::Spanish;
  }
  bool uses_german_rules() const { return rule_backend_ == RuleBasedG2pKind::German; }
  bool uses_french_rules() const { return rule_backend_ == RuleBasedG2pKind::French; }
  bool uses_dutch_rules() const { return rule_backend_ == RuleBasedG2pKind::Dutch; }
  bool uses_italian_rules() const { return rule_backend_ == RuleBasedG2pKind::Italian; }
  bool uses_russian_rules() const { return rule_backend_ == RuleBasedG2pKind::Russian; }
  bool uses_chinese_rules() const { return rule_backend_ == RuleBasedG2pKind::Chinese; }
  bool uses_portuguese_rules() const {
    return rule_backend_ == RuleBasedG2pKind::Portuguese;
  }
  bool uses_english_rules() const { return rule_backend_ == RuleBasedG2pKind::English; }
  /// Always false: full-bundle ONNX G2P was removed; English may still load optional heteronym/OOV
  /// ONNX inside ``EnglishRuleG2p``.
  static constexpr bool uses_onnx() { return false; }

  /// Canonical dialect id (e.g. ``es-AR`` for Spanish, ``en_us`` for US English).
  const std::string& dialect_id() const { return dialect_id_; }

 private:
  std::string dialect_id_;
  std::unique_ptr<RuleBasedG2p> rules_;
  std::optional<RuleBasedG2pKind> rule_backend_;
};

}  // namespace moonshine_g2p

#endif  // MOONSHINE_G2P_MOONSHINE_G2P_H
