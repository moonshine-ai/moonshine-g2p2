#ifndef MOONSHINE_G2P_MOONSHINE_G2P_H
#define MOONSHINE_G2P_MOONSHINE_G2P_H

#include "moonshine-g2p/g2p-word-log.h"
#include "moonshine-g2p/lang-specific/dutch.h"
#include "moonshine-g2p/lang-specific/italian.h"
#include "moonshine-g2p/lang-specific/russian.h"
#include "moonshine-g2p/lang-specific/portuguese.h"
#include "moonshine-g2p/lang-specific/french.h"
#include "moonshine-g2p/lang-specific/german.h"
#include "moonshine-g2p/lang-specific/spanish.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_g2p {

class MoonshineOnnxG2p;

/// True if *dialect_id* maps to the built-in Spanish rule engine (e.g. ``es-MX``, ``es_ar``).
[[nodiscard]] bool dialect_resolves_to_spanish_rules(
    std::string_view dialect_id, bool spanish_narrow_obstruents = true);

/// Options for constructing MoonshineG2P (ONNX paths and Spanish rule toggles).
struct MoonshineG2POptions {
  std::filesystem::path model_root = "models";
  bool use_cuda = false;
  /// Used when the dialect resolves to rule-based Spanish G2P.
  bool spanish_with_stress = true;
  bool spanish_narrow_obstruents = true;
  /// German rule G2P (``de``, ``de-DE``, ``german``): ``<model_root>/de/dict.tsv`` (default ``models/de/dict.tsv``) if unset.
  std::optional<std::filesystem::path> german_dict_path;
  bool german_with_stress = true;
  bool german_vocoder_stress = true;
  /// French rule G2P (``fr``, ``fr-FR``, ``french``): lexicon under ``<model-root>/../data/fr/dict.tsv`` or ``<model-root>/fr/dict.tsv``.
  std::optional<std::filesystem::path> french_dict_path;
  std::optional<std::filesystem::path> french_csv_dir;
  bool french_with_stress = true;
  bool french_liaison = true;
  bool french_liaison_optional = true;
  bool french_oov_rules = true;
  bool french_expand_cardinal_digits = true;
  /// Dutch rule G2P (``nl``, ``nl-NL``, ``dutch``): ``<model-root>/../data/nl/dict.tsv`` or ``<model-root>/nl/dict.tsv``.
  std::optional<std::filesystem::path> dutch_dict_path;
  bool dutch_with_stress = true;
  bool dutch_vocoder_stress = true;
  bool dutch_expand_cardinal_digits = true;
  /// Italian rule G2P (``it``, ``it-IT``, ``italian``): ``<model-root>/../data/it/dict.tsv`` or ``<model-root>/it/dict.tsv``.
  std::optional<std::filesystem::path> italian_dict_path;
  bool italian_with_stress = true;
  bool italian_vocoder_stress = true;
  bool italian_expand_cardinal_digits = true;
  /// Russian rule G2P (``ru``, ``ru-RU``, ``russian``): ``<model-root>/../data/ru/dict.tsv`` or ``<model-root>/ru/dict.tsv``.
  std::optional<std::filesystem::path> russian_dict_path;
  bool russian_with_stress = true;
  bool russian_vocoder_stress = true;
  /// Portuguese rule G2P (``pt_br``, ``pt-br``, ``pt_pt``, ``portugal``, …): ``<model-root>/../data/pt_br/dict.tsv`` or ``pt_pt``.
  std::optional<std::filesystem::path> portuguese_dict_path;
  bool portuguese_with_stress = true;
  bool portuguese_vocoder_stress = true;
  bool portuguese_keep_syllable_dots = false;
  bool portuguese_expand_cardinal_digits = true;
  bool portuguese_apply_pt_pt_final_esh = true;
  /// If set, override paths from g2p-config.json (same semantics as the CLI).
  std::optional<std::filesystem::path> dict_path_override;
  std::optional<std::filesystem::path> heteronym_onnx_override;
  std::optional<std::filesystem::path> oov_onnx_override;
};

/// Single entry point: *dialect_id* is a tag such as ``es-AR``, ``de``, ``de-DE``, or ``en_us``.
/// Rule-based engines are used when implemented (Spanish, German, French, Dutch, Italian, Russian, Portuguese); otherwise ONNX models under
/// ``model_root`` / ``<dialect_with_underscores>`` / ``g2p-config.json`` are loaded.
class MoonshineG2P {
 public:
  explicit MoonshineG2P(std::string dialect_id, MoonshineG2POptions options = {});
  ~MoonshineG2P();

  MoonshineG2P(const MoonshineG2P&) = delete;
  MoonshineG2P& operator=(const MoonshineG2P&) = delete;
  MoonshineG2P(MoonshineG2P&&) noexcept;
  MoonshineG2P& operator=(MoonshineG2P&&) noexcept;

  [[nodiscard]] std::string text_to_ipa(std::string_view text,
                                        std::vector<G2pWordLog>* per_word_log = nullptr);

  [[nodiscard]] bool uses_spanish_rules() const { return spanish_.has_value(); }
  [[nodiscard]] bool uses_german_rules() const { return german_.has_value(); }
  [[nodiscard]] bool uses_french_rules() const { return french_.has_value(); }
  [[nodiscard]] bool uses_dutch_rules() const { return dutch_.has_value(); }
  [[nodiscard]] bool uses_italian_rules() const { return italian_.has_value(); }
  [[nodiscard]] bool uses_russian_rules() const { return russian_.has_value(); }
  [[nodiscard]] bool uses_portuguese_rules() const { return portuguese_.has_value(); }
  [[nodiscard]] bool uses_onnx() const { return onnx_ != nullptr; }

  /// Canonical dialect id (e.g. ``es-AR`` for Spanish, or the normalized ONNX subdir form
  /// ``en_us``).
  [[nodiscard]] const std::string& dialect_id() const { return dialect_id_; }

 private:
  std::string dialect_id_;
  std::optional<SpanishDialect> spanish_;
  bool spanish_with_stress_ = true;
  std::optional<GermanRuleG2p> german_;
  std::optional<FrenchRuleG2p> french_;
  std::optional<DutchRuleG2p> dutch_;
  std::optional<ItalianRuleG2p> italian_;
  std::optional<RussianRuleG2p> russian_;
  std::optional<PortugueseRuleG2p> portuguese_;
  std::unique_ptr<MoonshineOnnxG2p> onnx_;
};

}  // namespace moonshine_g2p

#endif  // MOONSHINE_G2P_MOONSHINE_G2P_H
