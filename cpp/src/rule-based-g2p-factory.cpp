#include "moonshine-g2p/rule-based-g2p-factory.h"

#include "moonshine-g2p/moonshine-g2p-options.h"
#include "moonshine-g2p/rule-based-g2p.h"
#include "moonshine-g2p/lang-specific/dutch.h"
#include "moonshine-g2p/lang-specific/french.h"
#include "moonshine-g2p/lang-specific/german.h"
#include "moonshine-g2p/lang-specific/italian.h"
#include "moonshine-g2p/lang-specific/portuguese.h"
#include "moonshine-g2p/lang-specific/russian.h"
#include "moonshine-g2p/lang-specific/spanish.h"

#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <utility>

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

std::optional<RuleBasedG2pInstance> try_spanish(std::string_view trimmed,
                                                const MoonshineG2POptions& options) {
  const std::string spanish_key = normalize_spanish_dialect_cli_key(trimmed);
  try {
    SpanishDialect d =
        spanish_dialect_from_cli_id(spanish_key, options.spanish_narrow_obstruents);
    RuleBasedG2pInstance out;
    out.canonical_dialect_id = std::move(d.id);
    out.kind = RuleBasedG2pKind::Spanish;
    out.engine = std::make_unique<SpanishRuleG2p>(std::move(d), options.spanish_with_stress);
    return out;
  } catch (const std::invalid_argument&) {
    return std::nullopt;
  }
}

std::optional<RuleBasedG2pInstance> try_german(std::string_view trimmed,
                                               const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_german_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path gdict =
      options.german_dict_path.value_or(options.model_root / "de" / "dict.tsv");
  if (!std::filesystem::is_regular_file(gdict)) {
    throw std::runtime_error(
        "German G2P: lexicon not found at " + gdict.generic_string() +
        " (set MoonshineG2POptions::german_dict_path)");
  }
  GermanRuleG2p g(gdict, GermanRuleG2p::Options{.with_stress = options.german_with_stress,
                                                 .vocoder_stress = options.german_vocoder_stress});
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "de-DE";
  out.kind = RuleBasedG2pKind::German;
  out.engine = std::make_unique<GermanRuleG2p>(std::move(g));
  return out;
}

std::optional<RuleBasedG2pInstance> try_french(std::string_view trimmed,
                                               const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_french_rules(trimmed)) {
    return std::nullopt;
  }
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
  FrenchRuleG2p fr(fdict, fcsv, fo);
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "fr-FR";
  out.kind = RuleBasedG2pKind::French;
  out.engine = std::make_unique<FrenchRuleG2p>(std::move(fr));
  return out;
}

std::optional<RuleBasedG2pInstance> try_dutch(std::string_view trimmed,
                                              const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_dutch_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path ndict =
      options.dutch_dict_path.value_or(resolve_dutch_dict_path(options.model_root));
  if (!std::filesystem::is_regular_file(ndict)) {
    throw std::runtime_error(
        "Dutch G2P: lexicon not found at " + ndict.generic_string() +
        " (set MoonshineG2POptions::dutch_dict_path)");
  }
  DutchRuleG2p dutch(ndict, DutchRuleG2p::Options{.with_stress = options.dutch_with_stress,
                                                  .vocoder_stress = options.dutch_vocoder_stress,
                                                  .expand_cardinal_digits =
                                                      options.dutch_expand_cardinal_digits});
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "nl-NL";
  out.kind = RuleBasedG2pKind::Dutch;
  out.engine = std::make_unique<DutchRuleG2p>(std::move(dutch));
  return out;
}

std::optional<RuleBasedG2pInstance> try_italian(std::string_view trimmed,
                                                const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_italian_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path idict =
      options.italian_dict_path.value_or(resolve_italian_dict_path(options.model_root));
  if (!std::filesystem::is_regular_file(idict)) {
    throw std::runtime_error(
        "Italian G2P: lexicon not found at " + idict.generic_string() +
        " (set MoonshineG2POptions::italian_dict_path)");
  }
  ItalianRuleG2p it(idict, ItalianRuleG2p::Options{.with_stress = options.italian_with_stress,
                                                  .vocoder_stress = options.italian_vocoder_stress,
                                                  .expand_cardinal_digits =
                                                      options.italian_expand_cardinal_digits});
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "it-IT";
  out.kind = RuleBasedG2pKind::Italian;
  out.engine = std::make_unique<ItalianRuleG2p>(std::move(it));
  return out;
}

std::optional<RuleBasedG2pInstance> try_russian(std::string_view trimmed,
                                                const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_russian_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path rdict =
      options.russian_dict_path.value_or(resolve_russian_dict_path(options.model_root));
  if (!std::filesystem::is_regular_file(rdict)) {
    throw std::runtime_error(
        "Russian G2P: lexicon not found at " + rdict.generic_string() +
        " (set MoonshineG2POptions::russian_dict_path)");
  }
  RussianRuleG2p ru(rdict, RussianRuleG2p::Options{.with_stress = options.russian_with_stress,
                                                     .vocoder_stress = options.russian_vocoder_stress});
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "ru-RU";
  out.kind = RuleBasedG2pKind::Russian;
  out.engine = std::make_unique<RussianRuleG2p>(std::move(ru));
  return out;
}

std::optional<RuleBasedG2pInstance> try_portuguese(std::string_view trimmed,
                                                  const MoonshineG2POptions& options) {
  const bool want_pt_br = dialect_resolves_to_brazilian_portuguese_rules(trimmed);
  const bool want_pt_pt = dialect_resolves_to_portugal_rules(trimmed);
  if (!want_pt_br && !want_pt_pt) {
    return std::nullopt;
  }
  const bool is_portugal = want_pt_pt && !want_pt_br;
  const std::filesystem::path pdict =
      options.portuguese_dict_path.value_or(resolve_portuguese_dict_path(options.model_root, is_portugal));
  if (!std::filesystem::is_regular_file(pdict)) {
    throw std::runtime_error(
        "Portuguese G2P: lexicon not found at " + pdict.generic_string() +
        " (set MoonshineG2POptions::portuguese_dict_path)");
  }
  PortugueseRuleG2p pt(
      pdict, is_portugal,
      PortugueseRuleG2p::Options{.with_stress = options.portuguese_with_stress,
                                 .vocoder_stress = options.portuguese_vocoder_stress,
                                 .keep_syllable_dots = options.portuguese_keep_syllable_dots,
                                 .apply_pt_pt_final_esh = options.portuguese_apply_pt_pt_final_esh,
                                 .expand_cardinal_digits = options.portuguese_expand_cardinal_digits});
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = is_portugal ? "pt-PT" : "pt-BR";
  out.kind = RuleBasedG2pKind::Portuguese;
  out.engine = std::make_unique<PortugueseRuleG2p>(std::move(pt));
  return out;
}

using TryFn = std::optional<RuleBasedG2pInstance> (*)(std::string_view, const MoonshineG2POptions&);

const TryFn kTryChain[] = {
    try_spanish,
    try_german,
    try_french,
    try_dutch,
    try_italian,
    try_russian,
    try_portuguese,
};

}  // namespace

std::optional<RuleBasedG2pInstance> create_rule_based_g2p(std::string_view dialect_id,
                                                        const MoonshineG2POptions& options) {
  const std::string trimmed = trim_copy(dialect_id);
  if (trimmed.empty()) {
    throw std::invalid_argument("empty dialect id");
  }
  for (TryFn fn : kTryChain) {
    if (auto o = fn(trimmed, options)) {
      return o;
    }
  }
  return std::nullopt;
}

std::vector<std::pair<RuleBasedG2pKind, std::vector<std::string>>> rule_based_g2p_dialect_catalog() {
  std::vector<std::pair<RuleBasedG2pKind, std::vector<std::string>>> out;
  out.emplace_back(RuleBasedG2pKind::Spanish, SpanishRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::German, GermanRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::French, FrenchRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Dutch, DutchRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Italian, ItalianRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Russian, RussianRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Portuguese, PortugueseRuleG2p::dialect_ids());
  return out;
}

}  // namespace moonshine_g2p
