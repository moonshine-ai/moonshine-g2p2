#include "moonshine-g2p/moonshine-g2p.h"
#include "moonshine-g2p/rule-based-g2p-factory.h"
#include "moonshine-g2p/rule-based-g2p.h"
#include "moonshine-g2p/lang-specific/dutch.h"
#include "moonshine-g2p/lang-specific/english.h"
#include "moonshine-g2p/lang-specific/french.h"
#include "moonshine-g2p/lang-specific/german.h"
#include "moonshine-g2p/lang-specific/chinese.h"
#include "moonshine-g2p/lang-specific/italian.h"
#include "moonshine-g2p/lang-specific/portuguese.h"
#include "moonshine-g2p/lang-specific/russian.h"
#include "moonshine-g2p/lang-specific/spanish.h"

#include <cctype>
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

bool dialect_uses_rule_based_g2p(std::string_view dialect_id, const MoonshineG2POptions& options) {
  const std::string trimmed = trim_copy(dialect_id);
  if (trimmed.empty()) {
    return false;
  }
  if (dialect_resolves_to_english_rules(trimmed)) {
    return true;
  }
  if (dialect_resolves_to_spanish_rules(trimmed, options.spanish_narrow_obstruents)) {
    return true;
  }
  if (dialect_resolves_to_german_rules(trimmed)) {
    return true;
  }
  if (dialect_resolves_to_french_rules(trimmed)) {
    return true;
  }
  if (dialect_resolves_to_dutch_rules(trimmed)) {
    return true;
  }
  if (dialect_resolves_to_italian_rules(trimmed)) {
    return true;
  }
  if (dialect_resolves_to_russian_rules(trimmed)) {
    return true;
  }
  if (dialect_resolves_to_chinese_rules(trimmed)) {
    return true;
  }
  if (dialect_resolves_to_brazilian_portuguese_rules(trimmed) || dialect_resolves_to_portugal_rules(trimmed)) {
    return true;
  }
  return false;
}

MoonshineG2P::~MoonshineG2P() = default;

MoonshineG2P::MoonshineG2P(MoonshineG2P&&) noexcept = default;
MoonshineG2P& MoonshineG2P::operator=(MoonshineG2P&&) noexcept = default;

MoonshineG2P::MoonshineG2P(std::string dialect_id, MoonshineG2POptions options) {
  const std::string trimmed = trim_copy(dialect_id);
  if (trimmed.empty()) {
    throw std::invalid_argument("empty dialect id");
  }

  if (auto rb = create_rule_based_g2p(trimmed, options)) {
    dialect_id_ = std::move(rb->canonical_dialect_id);
    rules_ = std::move(rb->engine);
    rule_backend_ = rb->kind;
    return;
  }

  throw std::runtime_error(
      "MoonshineG2P: unsupported dialect \"" + trimmed +
      "\". Only rule-based locales are supported (e.g. en_us, es-MX, de, fr, nl, it, ru, zh, pt_br); "
      "see dialect_uses_rule_based_g2p() and rule_based_g2p_dialect_catalog().");
}

std::string MoonshineG2P::text_to_ipa(std::string_view text, std::vector<G2pWordLog>* per_word_log) {
  if (rules_) {
    return rules_->text_to_ipa(std::string(text), per_word_log);
  }
  throw std::logic_error("MoonshineG2P: no backend initialized");
}

}  // namespace moonshine_g2p
