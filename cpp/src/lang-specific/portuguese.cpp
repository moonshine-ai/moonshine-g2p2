#include "moonshine_g2p/lang-specific/portuguese.hpp"

#include "moonshine_g2p/g2p_word_log.hpp"
#include "moonshine_g2p/lang-specific/german.hpp"
#include "moonshine_g2p/utf8_utils.hpp"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cstdint>
#include <cwctype>
#include <fstream>
#include <limits>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace moonshine_g2p {
namespace {

const std::string kPri{"\xCB\x88"};
const std::string kSec{"\xCB\x8C"};

std::string trim_sv(std::string_view s) {
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

bool utf8_decode_at(const std::string& s, size_t i, char32_t& out_cp, size_t& out_len) {
  const size_t n = s.size();
  if (i >= n) {
    return false;
  }
  const unsigned char c0 = static_cast<unsigned char>(s[i]);
  if (c0 < 0x80) {
    out_cp = c0;
    out_len = 1;
    return true;
  }
  if ((c0 >> 5) == 0x6 && i + 1 < n) {
    const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
    if ((c1 >> 6) != 0x2) {
      out_cp = c0;
      out_len = 1;
      return true;
    }
    out_cp = (static_cast<char32_t>(c0 & 0x1Fu) << 6) | (c1 & 0x3Fu);
    out_len = 2;
    return true;
  }
  if ((c0 >> 4) == 0xE && i + 2 < n) {
    const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
    const unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
    if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2) {
      out_cp = c0;
      out_len = 1;
      return true;
    }
    out_cp = (static_cast<char32_t>(c0 & 0x0Fu) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    out_len = 3;
    return true;
  }
  if ((c0 >> 3) == 0x1E && i + 3 < n) {
    const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
    const unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
    const unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
    if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2 || (c3 >> 6) != 0x2) {
      out_cp = c0;
      out_len = 1;
      return true;
    }
    out_cp = (static_cast<char32_t>(c0 & 0x07u) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) |
             (c3 & 0x3F);
    out_len = 4;
    return true;
  }
  out_cp = c0;
  out_len = 1;
  return true;
}

char32_t pt_tolower(char32_t c) {
  switch (c) {
  case U'À':
    return U'à';
  case U'Á':
    return U'á';
  case U'Â':
    return U'â';
  case U'Ã':
    return U'ã';
  case U'Ç':
    return U'ç';
  case U'É':
    return U'é';
  case U'Ê':
    return U'ê';
  case U'Í':
    return U'í';
  case U'Ó':
    return U'ó';
  case U'Ô':
    return U'ô';
  case U'Õ':
    return U'õ';
  case U'Ú':
    return U'ú';
  case U'Ü':
    return U'ü';
  case U'Ý':
    return U'ý';
  default:
    break;
  }
  if (c >= U'A' && c <= U'Z') {
    return c + 32;
  }
  if ((c >= U'\u00C0' && c <= U'\u00D6') || (c >= U'\u00D8' && c <= U'\u00DE')) {
    return c + 32;
  }
  return c;
}

bool is_pt_key_cp(char32_t c) {
  c = pt_tolower(c);
  if (c == U'\u2019') {
    c = U'\'';
  }
  if (c >= U'a' && c <= U'z') {
    return true;
  }
  if (c == U'\'' || c == U'-') {
    return true;
  }
  return c == U'à' || c == U'á' || c == U'â' || c == U'ã' || c == U'ç' || c == U'é' || c == U'ê' ||
         c == U'í' || c == U'ó' || c == U'ô' || c == U'õ' || c == U'ú' || c == U'ü' || c == U'ý';
}

std::string normalize_lookup_key_utf8(const std::string& word) {
  std::string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word, i, cp, adv);
    if (cp == U'\u2019') {
      cp = U'\'';
    }
    const char32_t cl = pt_tolower(cp);
    if (is_pt_key_cp(cl)) {
      utf8_append_codepoint(out, cl);
    }
    i += adv;
  }
  return out;
}

std::string utf8_lowercase_pt_surface(const std::string& word) {
  std::string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word, i, cp, adv);
    utf8_append_codepoint(out, pt_tolower(cp));
    i += adv;
  }
  return out;
}

void load_pt_lexicon_file(const std::filesystem::path& path, std::unordered_map<std::string, std::string>& out) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Portuguese G2P: cannot read lexicon " + path.generic_string());
  }
  std::unordered_map<std::string, std::pair<std::string, bool>> tmp;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string surf = trim_sv(line.substr(0, tab));
    std::string ipa = trim_sv(line.substr(tab + 1));
    const std::string k = normalize_lookup_key_utf8(surf);
    if (k.empty()) {
      continue;
    }
    const bool lower_ok = (surf == utf8_lowercase_pt_surface(surf));
    const auto it = tmp.find(k);
    if (it == tmp.end()) {
      tmp.emplace(k, std::make_pair(std::move(ipa), lower_ok));
    } else if (lower_ok && !it->second.second) {
      it->second = std::make_pair(std::move(ipa), true);
    }
  }
  out.clear();
  for (auto& e : tmp) {
    out.emplace(std::move(e.first), std::move(e.second.first));
  }
}

void erase_utf8_substr(std::string& s, const std::string& sub) {
  size_t p = 0;
  while ((p = s.find(sub, p)) != std::string::npos) {
    s.erase(p, sub.size());
  }
}

// --- Numbers (portuguese_numbers.py) ------------------------------------------

bool is_all_ascii_digits(std::string_view s) {
  if (s.empty()) {
    return false;
  }
  for (char c : s) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  return true;
}

static const char* kDigPt[10] = {"zero", "um", "dois", "tr\xc3\xaas", "quatro", "cinco", "seis", "sete", "oito",
                                 "nove"};

static const char* kTensPt[10] = {"", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta",
                                    "oitenta", "noventa"};

static const char* kHundredsPt[10] = {"", "", "duzentos", "trezentos", "quatrocentos", "quinhentos", "seiscentos",
                                        "setecentos", "oitocentos", "novecentos"};

std::string teens_word_pt(int n, bool is_pt_pt) {
  if (n < 11 || n > 19) {
    throw std::out_of_range("teens");
  }
  if (n == 18) {
    return "dezoito";
  }
  if (is_pt_pt) {
    switch (n) {
    case 11:
      return "onze";
    case 12:
      return "doze";
    case 13:
      return "treze";
    case 14:
      return "catorze";
    case 15:
      return "quinze";
    case 16:
      return "dezasseis";
    case 17:
      return "dezassete";
    case 19:
      return "dezanove";
    default:
      break;
    }
  }
  switch (n) {
  case 11:
    return "onze";
  case 12:
    return "doze";
  case 13:
    return "treze";
  case 14:
    return "catorze";
  case 15:
    return "quinze";
  case 16:
    return "dezesseis";
  case 17:
    return "dezessete";
  case 19:
    return "dezenove";
  default:
    break;
  }
  throw std::out_of_range("teens");
}

void under_100_tokens_pt(int n, bool is_pt_pt, std::vector<std::string>& out) {
  if (n < 0 || n >= 100) {
    throw std::out_of_range("under_100");
  }
  if (n < 10) {
    out.emplace_back(kDigPt[n]);
    return;
  }
  if (n == 10) {
    out.emplace_back("dez");
    return;
  }
  if (n < 20) {
    out.emplace_back(teens_word_pt(n, is_pt_pt));
    return;
  }
  const int t = n / 10;
  const int u = n % 10;
  if (u == 0) {
    out.emplace_back(kTensPt[t]);
    return;
  }
  out.emplace_back(kTensPt[t]);
  out.emplace_back("e");
  out.emplace_back(kDigPt[u]);
}

void below_1000_tokens_pt(int n, bool is_pt_pt, std::vector<std::string>& out) {
  if (n < 0 || n >= 1000) {
    throw std::out_of_range("below_1000");
  }
  if (n < 100) {
    under_100_tokens_pt(n, is_pt_pt, out);
    return;
  }
  const int h = n / 100;
  const int r = n % 100;
  if (h == 1) {
    if (r == 0) {
      out.emplace_back("cem");
      return;
    }
    out.emplace_back("cento");
    out.emplace_back("e");
    under_100_tokens_pt(r, is_pt_pt, out);
    return;
  }
  out.emplace_back(kHundredsPt[h]);
  if (r == 0) {
    return;
  }
  out.emplace_back("e");
  under_100_tokens_pt(r, is_pt_pt, out);
}

void below_1_000_000_tokens_pt(int n, bool is_pt_pt, std::vector<std::string>& out) {
  if (n < 0 || n >= 1'000'000) {
    throw std::out_of_range("below_1m");
  }
  if (n < 1000) {
    below_1000_tokens_pt(n, is_pt_pt, out);
    return;
  }
  const int q = n / 1000;
  const int r = n % 1000;
  if (q == 1) {
    out.emplace_back("mil");
  } else {
    below_1000_tokens_pt(q, is_pt_pt, out);
    out.emplace_back("mil");
  }
  if (r == 0) {
    return;
  }
  out.emplace_back("e");
  below_1000_tokens_pt(r, is_pt_pt, out);
}

std::string expand_cardinal_digits_to_portuguese_words(std::string_view s, bool is_pt_pt) {
  if (!is_all_ascii_digits(s)) {
    return std::string(s);
  }
  if (s.size() > 1 && s[0] == '0') {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) {
        o.push_back(' ');
      }
      o += kDigPt[static_cast<size_t>(s[i] - '0')];
    }
    return o;
  }
  long long n = 0;
  for (char c : s) {
    n = n * 10 + (c - '0');
  }
  if (n > 999'999) {
    return std::string(s);
  }
  if (n == 0) {
    return "zero";
  }
  std::vector<std::string> parts;
  below_1_000_000_tokens_pt(static_cast<int>(n), is_pt_pt, parts);
  std::string o;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      o.push_back(' ');
    }
    o += parts[i];
  }
  return o;
}

std::string expand_digit_tokens_in_text(std::string text, bool is_pt_pt) {
  static const std::regex range_re(R"(\b(\d+)-(\d+)\b)", std::regex::ECMAScript);
  static const std::regex dig_re(R"(\b\d+\b)", std::regex::ECMAScript);
  std::string out;
  std::sregex_iterator it(text.begin(), text.end(), range_re);
  std::sregex_iterator end;
  size_t pos = 0;
  for (; it != end; ++it) {
    const std::smatch& m = *it;
    out.append(text, pos, static_cast<size_t>(m.position() - static_cast<long>(pos)));
    out += expand_cardinal_digits_to_portuguese_words(m.str(1), is_pt_pt);
    out += " - ";
    out += expand_cardinal_digits_to_portuguese_words(m.str(2), is_pt_pt);
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  text = std::move(out);
  out.clear();
  pos = 0;
  std::sregex_iterator it2(text.begin(), text.end(), dig_re);
  for (; it2 != end; ++it2) {
    const std::smatch& m = *it2;
    out.append(text, pos, static_cast<size_t>(m.position() - static_cast<long>(pos)));
    out += expand_cardinal_digits_to_portuguese_words(m.str(), is_pt_pt);
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  return out;
}

#include "lang-specific/portuguese_rules.inc"

bool is_pt_word_char(char32_t cp) {
  if (cp == U'-' || cp == U'\'' || cp == U'\u2019') {
    return true;
  }
  if (cp >= U'0' && cp <= U'9') {
    return true;
  }
  if (cp == U'_') {
    return true;
  }
  if (cp <= 0x7Fu) {
    return std::isalpha(static_cast<unsigned char>(cp)) != 0;
  }
  static const bool utf8_ctype = []() {
    return std::setlocale(LC_CTYPE, "C.UTF-8") != nullptr || std::setlocale(LC_CTYPE, "en_US.UTF-8") != nullptr;
  }();
  if (utf8_ctype && cp <= static_cast<char32_t>(std::numeric_limits<wint_t>::max())) {
    const wint_t w = static_cast<wint_t>(cp);
    if (std::iswalpha(w) != 0 || std::iswdigit(w) != 0) {
      return true;
    }
  }
  cp = pt_tolower(cp);
  if (cp >= U'a' && cp <= U'z') {
    return true;
  }
  return cp == U'à' || cp == U'á' || cp == U'â' || cp == U'ã' || cp == U'ç' || cp == U'é' || cp == U'ê' ||
         cp == U'í' || cp == U'ó' || cp == U'ô' || cp == U'õ' || cp == U'ú' || cp == U'ü' || cp == U'ý';
}

bool try_consume_pt_word(const std::string& text, size_t pos, size_t& out_end) {
  if (pos >= text.size()) {
    return false;
  }
  char32_t c0 = 0;
  size_t a0 = 0;
  utf8_decode_at(text, pos, c0, a0);
  if (c0 == U'\'' || c0 == U'\u2019') {
    if (pos + a0 >= text.size()) {
      return false;
    }
    char32_t c1 = 0;
    size_t a1 = 0;
    utf8_decode_at(text, pos + a0, c1, a1);
    if (!is_pt_word_char(c1) || c1 == U'\'' || c1 == U'\u2019') {
      return false;
    }
  } else if (!is_pt_word_char(c0) || c0 == U'\'' || c0 == U'\u2019') {
    return false;
  }
  size_t p = pos;
  while (p < text.size()) {
    char32_t c = 0;
    size_t a = 0;
    utf8_decode_at(text, p, c, a);
    if (is_pt_word_char(c)) {
      p += a;
      continue;
    }
    if (c == U'\'' || c == U'\u2019') {
      if (p + a >= text.size()) {
        p += a;
        break;
      }
      char32_t c2 = 0;
      size_t a2 = 0;
      utf8_decode_at(text, p + a, c2, a2);
      if (is_pt_word_char(c2) && c2 != U'\'' && c2 != U'\u2019') {
        p += a + a2;
        continue;
      }
      p += a;
      break;
    }
    break;
  }
  out_end = p;
  return out_end > pos;
}

}  // namespace

PortugueseRuleG2p::PortugueseRuleG2p(std::filesystem::path dict_tsv, bool is_portugal)
    : PortugueseRuleG2p(std::move(dict_tsv), is_portugal, Options{}) {}

PortugueseRuleG2p::PortugueseRuleG2p(std::filesystem::path dict_tsv, bool is_portugal, Options options)
    : is_portugal_(is_portugal), options_(options) {
  dialect_id_ = is_portugal ? "pt-PT" : "pt-BR";
  load_pt_lexicon_file(std::move(dict_tsv), lexicon_);
}

std::string PortugueseRuleG2p::finalize_ipa(std::string ipa, bool from_lexicon) const {
  if (!options_.with_stress) {
    erase_utf8_substr(ipa, kPri);
    erase_utf8_substr(ipa, kSec);
    if (!options_.keep_syllable_dots) {
      ipa.erase(std::remove(ipa.begin(), ipa.end(), '.'), ipa.end());
    }
    return ipa;
  }
  if (options_.vocoder_stress && !from_lexicon) {
    ipa = GermanRuleG2p::normalize_ipa_stress_for_vocoder(std::move(ipa));
  }
  if (!options_.keep_syllable_dots) {
    ipa.erase(std::remove(ipa.begin(), ipa.end(), '.'), ipa.end());
  }
  return ipa;
}

std::string PortugueseRuleG2p::lookup_or_rules(const std::string& raw_word) const {
  const std::string letters_only = normalize_lookup_key_utf8(raw_word);
  if (letters_only.empty()) {
    return "";
  }
  const auto rom = roman_numeral_token_to_ipa(letters_only, is_portugal_);
  if (rom.has_value()) {
    return finalize_ipa(*rom, true);
  }
  auto it = lexicon_.find(letters_only);
  if (it != lexicon_.end()) {
    return finalize_ipa(it->second, true);
  }
  if (letters_only.find('-') != std::string::npos) {
    std::vector<std::string> chunks;
    size_t start = 0;
    while (start < letters_only.size()) {
      size_t dash = letters_only.find('-', start);
      const size_t end = dash == std::string::npos ? letters_only.size() : dash;
      if (end > start) {
        chunks.push_back(letters_only.substr(start, end - start));
      }
      if (dash == std::string::npos) {
        break;
      }
      start = dash + 1;
    }
    bool ok = true;
    std::string merged;
    for (size_t c = 0; c < chunks.size(); ++c) {
      auto hc = lexicon_.find(chunks[c]);
      if (hc == lexicon_.end()) {
        ok = false;
        break;
      }
      if (c > 0) {
        merged += '-';
      }
      merged += hc->second;
    }
    if (ok) {
      return finalize_ipa(std::move(merged), true);
    }
  }
  const auto& fw = is_portugal_ ? kFwPt : kFwBr;
  auto fi = fw.find(letters_only);
  if (fi != fw.end()) {
    return finalize_ipa(fi->second, false);
  }
  std::string ipa_rules = rules_word_to_ipa_utf8(trim_sv(raw_word), is_portugal_, options_.with_stress);
  const bool oov_sc = kScStraddle.count(letters_only) != 0;
  if (is_portugal_ && options_.apply_pt_pt_final_esh && !oov_sc) {
    ipa_rules = pt_pt_apply_rules_final_s_to_esh(std::move(ipa_rules), letters_only);
  }
  return finalize_ipa(std::move(ipa_rules), oov_sc);
}

std::string PortugueseRuleG2p::word_to_ipa(const std::string& word) const {
  const std::string wraw = trim_sv(word);
  if (wraw.empty()) {
    return "";
  }
  if (options_.expand_cardinal_digits && is_all_ascii_digits(wraw)) {
    const std::string phrase = expand_cardinal_digits_to_portuguese_words(wraw, is_portugal_);
    if (phrase != wraw) {
      return text_to_ipa_no_expand(phrase, nullptr);
    }
    return wraw;
  }
  if (!options_.expand_cardinal_digits) {
    static const std::regex dig_pass(R"(^[0-9]+(?:-[0-9]+)*$)", std::regex::ECMAScript);
    if (std::regex_match(wraw, dig_pass)) {
      return wraw;
    }
  }
  return lookup_or_rules(wraw);
}

std::string PortugueseRuleG2p::text_to_ipa_no_expand(const std::string& text,
                                                     std::vector<G2pWordLog>* per_word_log) const {
  std::string out;
  size_t pos = 0;
  const size_t n = text.size();
  while (pos < n) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(text, pos, cp, adv);
    if (cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') {
      out.push_back(' ');
      while (pos < n) {
        utf8_decode_at(text, pos, cp, adv);
        if (cp != U' ' && cp != U'\t' && cp != U'\n' && cp != U'\r') {
          break;
        }
        pos += adv;
      }
      continue;
    }
    size_t wend = 0;
    if (try_consume_pt_word(text, pos, wend)) {
      const std::string tok = text.substr(pos, wend - pos);
      const std::string k = normalize_lookup_key_utf8(tok);
      std::string wipa = word_to_ipa(tok);
      if (per_word_log != nullptr) {
        per_word_log->push_back(G2pWordLog{tok, k, G2pWordPath::kRuleBasedG2p, wipa});
        out += per_word_log->back().ipa;
      } else {
        out += wipa;
      }
      pos = wend;
      continue;
    }
    const size_t start = pos;
    pos += adv;
    while (pos < n) {
      utf8_decode_at(text, pos, cp, adv);
      size_t tmp = 0;
      if (try_consume_pt_word(text, pos, tmp) || cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') {
        break;
      }
      pos += adv;
    }
    out.append(text, start, pos - start);
  }
  std::string collapsed;
  bool in_sp = false;
  for (char c : out) {
    if (c == ' ') {
      if (!in_sp && !collapsed.empty()) {
        collapsed.push_back(' ');
      }
      in_sp = true;
    } else {
      in_sp = false;
      collapsed.push_back(c);
    }
  }
  while (!collapsed.empty() && collapsed.front() == ' ') {
    collapsed.erase(collapsed.begin());
  }
  while (!collapsed.empty() && collapsed.back() == ' ') {
    collapsed.pop_back();
  }
  return collapsed;
}

std::string PortugueseRuleG2p::text_to_ipa(const std::string& text,
                                           std::vector<G2pWordLog>* per_word_log) const {
  std::string work = text;
  if (options_.expand_cardinal_digits) {
    work = expand_digit_tokens_in_text(std::move(work), is_portugal_);
  }
  return text_to_ipa_no_expand(work, per_word_log);
}

static std::string norm_pt_dialect_key(std::string_view raw) {
  std::string s(trim_sv(raw));
  for (char& c : s) {
    if (c == '_') {
      c = '-';
    }
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

bool dialect_resolves_to_portugal_rules(std::string_view dialect_id) {
  const std::string s = norm_pt_dialect_key(dialect_id);
  return s == "pt-pt" || s == "pt_pt" || s == "portugal" || s == "european-portuguese" ||
         s == "europeanportuguese";
}

bool dialect_resolves_to_brazilian_portuguese_rules(std::string_view dialect_id) {
  const std::string s = norm_pt_dialect_key(dialect_id);
  return s == "pt-br" || s == "pt_br" || s == "brazil" || s == "brazilian-portuguese" || s == "brazilianportuguese" ||
         s == "portuguese-brazil";
}

std::filesystem::path resolve_portuguese_dict_path(const std::filesystem::path& model_root, bool is_portugal) {
  const char* sub = is_portugal ? "pt_pt" : "pt_br";
  const std::filesystem::path under_data = model_root.parent_path() / "data" / sub / "dict.tsv";
  if (std::filesystem::is_regular_file(under_data)) {
    return under_data;
  }
  return model_root / sub / "dict.tsv";
}

}  // namespace moonshine_g2p
