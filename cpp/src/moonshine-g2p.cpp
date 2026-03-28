#include "moonshine-g2p/moonshine-g2p.h"
#include "moonshine-g2p/moonshine-onnx-g2p.h"

#include <cctype>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
namespace moonshine_g2p {

namespace {

std::string trim_copy(std::string_view s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a])) != 0) {
    ++a;
  }
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])) != 0) {
    --b;
  }
  return std::string(s.substr(a, b - a));
}

std::filesystem::path resolve_french_dict_path(const MoonshineG2POptions& opt) {
  if (opt.french_dict_path) {
    return *opt.french_dict_path;
  }
  const std::filesystem::path under_model = opt.model_root / "fr" / "dict.tsv";
  if (std::filesystem::is_regular_file(under_model)) {
    return under_model;
  }
  return opt.model_root.parent_path() / "data" / "fr" / "dict.tsv";
}

std::filesystem::path resolve_french_csv_dir(const MoonshineG2POptions& opt) {
  if (opt.french_csv_dir) {
    return *opt.french_csv_dir;
  }
  const std::filesystem::path data_fr = opt.model_root.parent_path() / "data" / "fr";
  if (std::filesystem::is_directory(data_fr)) {
    return data_fr;
  }
  return opt.model_root / "fr";
}

/// Normalize user input like ``es_ar`` / ``es-mx`` to keys accepted by
/// ``spanish_dialect_from_cli_id`` (e.g. ``es-AR``, ``es-MX``).
std::string normalize_spanish_dialect_cli_key(std::string_view raw) {
  std::string s = trim_copy(raw);
  for (char& c : s) {
    if (c == '_') {
      c = '-';
    }
  }
  if (s.size() >= 3 && (s[0] == 'e' || s[0] == 'E') && (s[1] == 's' || s[1] == 'S') &&
      s[2] == '-') {
    s[0] = 'e';
    s[1] = 's';
    size_t i = 3;
    while (i < s.size() && s[i] != '-') {
      if (std::isalpha(static_cast<unsigned char>(s[i])) != 0) {
        s[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
      }
      ++i;
    }
    if (i < s.size() && s[i] == '-') {
      ++i;
      while (i < s.size()) {
        s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
        ++i;
      }
    }
  }
  return s;
}

/// Directory name under ``models/`` (e.g. ``en_us``).
std::string dialect_to_onnx_model_subdir(std::string_view raw) {
  std::string s = trim_copy(raw);
  for (char& c : s) {
    if (c == '-') {
      c = '_';
    } else {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
  }
  return s;
}

void resolve_onnx_paths(const std::filesystem::path& data_root, const MoonshineG2POptions& opt,
                        std::optional<std::filesystem::path>& dict_path,
                        std::optional<std::filesystem::path>& het_onnx_path,
                        std::optional<std::filesystem::path>& oov_onnx_path) {
  const std::filesystem::path g2p_config_path = data_root / "g2p-config.json";
  const bool g2p_config_exists = std::filesystem::is_regular_file(g2p_config_path);
  if (!g2p_config_exists) {
    if (!opt.heteronym_onnx_override && !opt.oov_onnx_override && !opt.dict_path_override) {
      throw std::runtime_error(
          "No ONNX G2P bundle: g2p-config.json not found at " + g2p_config_path.generic_string() +
          " (use path overrides or pick a dialect with rule-based G2P, e.g. es-MX, pt_br, de).");
    }
    dict_path = opt.dict_path_override;
    het_onnx_path = opt.heteronym_onnx_override;
    oov_onnx_path = opt.oov_onnx_override;
    return;
  }
  std::ifstream g2p_config_in(g2p_config_path);
  const nlohmann::json g2p_config = nlohmann::json::parse(g2p_config_in);
  if (g2p_config["uses_dictionary"].get<bool>()) {
    dict_path = data_root / "dict_filtered_heteronyms.tsv";
  }
  if (g2p_config["uses_heteronym_model"].get<bool>()) {
    het_onnx_path = data_root / "heteronym" / "model.onnx";
  }
  if (g2p_config["uses_oov_model"].get<bool>()) {
    oov_onnx_path = data_root / "oov" / "model.onnx";
  }
}

void require_exists_if_set(const std::optional<std::filesystem::path>& p, const char* label) {
  if (!p) {
    return;
  }
  if (!std::filesystem::is_regular_file(*p)) {
    throw std::runtime_error(std::string(label) + " not found at " + p->generic_string());
  }
}

}  // namespace

bool dialect_resolves_to_spanish_rules(std::string_view dialect_id, bool spanish_narrow_obstruents) {
  const std::string trimmed = trim_copy(dialect_id);
  if (trimmed.empty()) {
    return false;
  }
  const std::string spanish_key = normalize_spanish_dialect_cli_key(trimmed);
  try {
    (void)spanish_dialect_from_cli_id(spanish_key, spanish_narrow_obstruents);
    return true;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

MoonshineG2P::~MoonshineG2P() = default;

MoonshineG2P::MoonshineG2P(MoonshineG2P&&) noexcept = default;
MoonshineG2P& MoonshineG2P::operator=(MoonshineG2P&&) noexcept = default;

MoonshineG2P::MoonshineG2P(std::string dialect_id, MoonshineG2POptions options) {
  const std::string trimmed = trim_copy(dialect_id);
  if (trimmed.empty()) {
    throw std::invalid_argument("empty dialect id");
  }

  const std::string spanish_key = normalize_spanish_dialect_cli_key(trimmed);
  try {
    SpanishDialect d =
        spanish_dialect_from_cli_id(spanish_key, options.spanish_narrow_obstruents);
    dialect_id_ = std::move(d.id);
    spanish_ = std::move(d);
    spanish_with_stress_ = options.spanish_with_stress;
    return;
  } catch (const std::invalid_argument&) {
  }

  if (dialect_resolves_to_german_rules(trimmed)) {
    const std::filesystem::path gdict =
        options.german_dict_path.value_or(options.model_root / "de" / "dict.tsv");
    if (!std::filesystem::is_regular_file(gdict)) {
      throw std::runtime_error(
          "German G2P: lexicon not found at " + gdict.generic_string() +
          " (set MoonshineG2POptions::german_dict_path)");
    }
    german_.emplace(gdict, GermanRuleG2p::Options{.with_stress = options.german_with_stress,
                                                  .vocoder_stress = options.german_vocoder_stress});
    dialect_id_ = "de-DE";
    return;
  }

  if (dialect_resolves_to_french_rules(trimmed)) {
    const std::filesystem::path fdict = resolve_french_dict_path(options);
    if (!std::filesystem::is_regular_file(fdict)) {
      throw std::runtime_error(
          "French G2P: lexicon not found at " + fdict.generic_string() +
          " (set MoonshineG2POptions::french_dict_path)");
    }
    const std::filesystem::path fcsv = resolve_french_csv_dir(options);
    FrenchRuleG2p::Options fo;
    fo.with_stress = options.french_with_stress;
    fo.liaison = options.french_liaison;
    fo.liaison_optional = options.french_liaison_optional;
    fo.oov_rules = options.french_oov_rules;
    fo.expand_cardinal_digits = options.french_expand_cardinal_digits;
    french_.emplace(fdict, fcsv, fo);
    dialect_id_ = "fr-FR";
    return;
  }

  if (dialect_resolves_to_dutch_rules(trimmed)) {
    const std::filesystem::path ndict =
        options.dutch_dict_path.value_or(resolve_dutch_dict_path(options.model_root));
    if (!std::filesystem::is_regular_file(ndict)) {
      throw std::runtime_error(
          "Dutch G2P: lexicon not found at " + ndict.generic_string() +
          " (set MoonshineG2POptions::dutch_dict_path)");
    }
    dutch_.emplace(ndict, DutchRuleG2p::Options{.with_stress = options.dutch_with_stress,
                                                  .vocoder_stress = options.dutch_vocoder_stress,
                                                  .expand_cardinal_digits =
                                                      options.dutch_expand_cardinal_digits});
    dialect_id_ = "nl-NL";
    return;
  }

  if (dialect_resolves_to_italian_rules(trimmed)) {
    const std::filesystem::path idict =
        options.italian_dict_path.value_or(resolve_italian_dict_path(options.model_root));
    if (!std::filesystem::is_regular_file(idict)) {
      throw std::runtime_error(
          "Italian G2P: lexicon not found at " + idict.generic_string() +
          " (set MoonshineG2POptions::italian_dict_path)");
    }
    italian_.emplace(idict, ItalianRuleG2p::Options{.with_stress = options.italian_with_stress,
                                                    .vocoder_stress = options.italian_vocoder_stress,
                                                    .expand_cardinal_digits =
                                                        options.italian_expand_cardinal_digits});
    dialect_id_ = "it-IT";
    return;
  }

  if (dialect_resolves_to_russian_rules(trimmed)) {
    const std::filesystem::path rdict =
        options.russian_dict_path.value_or(resolve_russian_dict_path(options.model_root));
    if (!std::filesystem::is_regular_file(rdict)) {
      throw std::runtime_error(
          "Russian G2P: lexicon not found at " + rdict.generic_string() +
          " (set MoonshineG2POptions::russian_dict_path)");
    }
    russian_.emplace(rdict, RussianRuleG2p::Options{.with_stress = options.russian_with_stress,
                                                      .vocoder_stress = options.russian_vocoder_stress});
    dialect_id_ = "ru-RU";
    return;
  }

  const bool want_pt_br = dialect_resolves_to_brazilian_portuguese_rules(trimmed);
  const bool want_pt_pt = dialect_resolves_to_portugal_rules(trimmed);
  if (want_pt_br || want_pt_pt) {
    const bool is_portugal = want_pt_pt && !want_pt_br;
    const std::filesystem::path pdict =
        options.portuguese_dict_path.value_or(resolve_portuguese_dict_path(options.model_root, is_portugal));
    if (!std::filesystem::is_regular_file(pdict)) {
      throw std::runtime_error(
          "Portuguese G2P: lexicon not found at " + pdict.generic_string() +
          " (set MoonshineG2POptions::portuguese_dict_path)");
    }
    portuguese_.emplace(
        pdict, is_portugal,
        PortugueseRuleG2p::Options{.with_stress = options.portuguese_with_stress,
                                    .vocoder_stress = options.portuguese_vocoder_stress,
                                    .keep_syllable_dots = options.portuguese_keep_syllable_dots,
                                    .apply_pt_pt_final_esh = options.portuguese_apply_pt_pt_final_esh,
                                    .expand_cardinal_digits = options.portuguese_expand_cardinal_digits});
    dialect_id_ = is_portugal ? "pt-PT" : "pt-BR";
    return;
  }

  dialect_id_ = dialect_to_onnx_model_subdir(trimmed);
  std::optional<std::filesystem::path> dict_path;
  std::optional<std::filesystem::path> het_onnx_path;
  std::optional<std::filesystem::path> oov_onnx_path;
  const std::filesystem::path data_root = options.model_root / dialect_id_;
  resolve_onnx_paths(data_root, options, dict_path, het_onnx_path, oov_onnx_path);

  require_exists_if_set(dict_path, "dictionary TSV");
  require_exists_if_set(het_onnx_path, "heteronym ONNX model");
  require_exists_if_set(oov_onnx_path, "OOV ONNX model");

  onnx_ = std::make_unique<MoonshineOnnxG2p>(dict_path, het_onnx_path, oov_onnx_path,
                                             options.use_cuda);
}

std::string MoonshineG2P::text_to_ipa(std::string_view text, std::vector<G2pWordLog>* per_word_log) {
  if (spanish_.has_value()) {
    return spanish_text_to_ipa(std::string(text), *spanish_, spanish_with_stress_, per_word_log);
  }
  if (german_.has_value()) {
    return german_->text_to_ipa(std::string(text), per_word_log);
  }
  if (french_.has_value()) {
    return french_->text_to_ipa(std::string(text), per_word_log);
  }
  if (dutch_.has_value()) {
    return dutch_->text_to_ipa(std::string(text), per_word_log);
  }
  if (italian_.has_value()) {
    return italian_->text_to_ipa(std::string(text), per_word_log);
  }
  if (russian_.has_value()) {
    return russian_->text_to_ipa(std::string(text), per_word_log);
  }
  if (portuguese_.has_value()) {
    return portuguese_->text_to_ipa(std::string(text), per_word_log);
  }
  if (onnx_) {
    return onnx_->text_to_ipa(text, per_word_log);
  }
  throw std::logic_error("MoonshineG2P: no backend initialized");
}

}  // namespace moonshine_g2p
