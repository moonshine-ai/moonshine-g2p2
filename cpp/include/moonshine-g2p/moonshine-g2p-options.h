#ifndef MOONSHINE_G2P_MOONSHINE_G2P_OPTIONS_H
#define MOONSHINE_G2P_MOONSHINE_G2P_OPTIONS_H

#include <filesystem>
#include <optional>

namespace moonshine_g2p {

/// Options for constructing ``MoonshineG2P`` (rule-engine paths and toggles; optional heteronym/OOV
/// ONNX overrides for English).
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
  /// English rule G2P (``en_us``, ``en-US``, …): ``<model-root>/en_us/dict_filtered_heteronyms.tsv``.
  std::optional<std::filesystem::path> english_dict_path;
  /// Override heteronym / OOV ONNX paths from ``en_us/g2p-config.json`` (English only).
  std::optional<std::filesystem::path> heteronym_onnx_override;
  std::optional<std::filesystem::path> oov_onnx_override;
};

}  // namespace moonshine_g2p

#endif  // MOONSHINE_G2P_MOONSHINE_G2P_OPTIONS_H
