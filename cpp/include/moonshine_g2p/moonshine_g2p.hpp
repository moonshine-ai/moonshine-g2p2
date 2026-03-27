#pragma once

#include "moonshine_g2p/g2p_word_log.hpp"
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
  /// If set, override paths from g2p-config.json (same semantics as the CLI).
  std::optional<std::filesystem::path> dict_path_override;
  std::optional<std::filesystem::path> heteronym_onnx_override;
  std::optional<std::filesystem::path> oov_onnx_override;
};

/// Single entry point: *dialect_id* is a tag such as ``es-AR``, ``de``, ``de-DE``, or ``en_us``.
/// Rule-based engines are used when implemented (Spanish, German); otherwise ONNX models under
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
  [[nodiscard]] bool uses_onnx() const { return onnx_ != nullptr; }

  /// Canonical dialect id (e.g. ``es-AR`` for Spanish, or the normalized ONNX subdir form
  /// ``en_us``).
  [[nodiscard]] const std::string& dialect_id() const { return dialect_id_; }

 private:
  std::string dialect_id_;
  std::optional<SpanishDialect> spanish_;
  bool spanish_with_stress_ = true;
  std::optional<GermanRuleG2p> german_;
  std::unique_ptr<MoonshineOnnxG2p> onnx_;
};

}  // namespace moonshine_g2p
