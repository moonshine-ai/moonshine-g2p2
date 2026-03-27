#pragma once

#include "moonshine_g2p/g2p_word_log.hpp"
#include "moonshine_g2p/lang-specific/dutch.hpp"
#include "moonshine_g2p/lang-specific/french.hpp"
#include "moonshine_g2p/lang-specific/german.hpp"
#include "moonshine_g2p/lang-specific/spanish.hpp"

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
  /// If set, override paths from g2p-config.json (same semantics as the CLI).
  std::optional<std::filesystem::path> dict_path_override;
  std::optional<std::filesystem::path> heteronym_onnx_override;
  std::optional<std::filesystem::path> oov_onnx_override;
};

/// Single entry point: *dialect_id* is a tag such as ``es-AR``, ``de``, ``de-DE``, or ``en_us``.
/// Rule-based engines are used when implemented (Spanish, German, French, Dutch); otherwise ONNX models under
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
  std::unique_ptr<MoonshineOnnxG2p> onnx_;
};

}  // namespace moonshine_g2p
