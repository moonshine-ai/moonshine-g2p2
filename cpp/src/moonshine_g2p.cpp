#include "moonshine_g2p/moonshine_g2p.hpp"
#include "moonshine_g2p/moonshine_onnx_g2p.hpp"

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
          " (use path overrides or pick a dialect with rule-based G2P, e.g. es-MX).");
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
  if (onnx_) {
    return onnx_->text_to_ipa(text, per_word_log);
  }
  throw std::logic_error("MoonshineG2P: no backend initialized");
}

}  // namespace moonshine_g2p
