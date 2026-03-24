#include "moonshine_g2p/ipa_postprocess.hpp"

#include "moonshine_g2p/utf8_utils.hpp"

#include <algorithm>
#include <cctype>

namespace moonshine_g2p {

namespace {

std::string trim_copy(std::string t) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  t.erase(t.begin(), std::find_if(t.begin(), t.end(), not_space));
  t.erase(std::find_if(t.rbegin(), t.rend(), not_space).base(), t.end());
  return t;
}

std::string strip_length_markers_copy(std::string t) {
  // Python: e1 = e0.replace("ː", "")  (U+02D0 MODIFIER LETTER TRIANGULAR COLON)
  const std::string mark = "\xCB\x90";  // UTF-8 for U+02D0
  size_t p = 0;
  while ((p = t.find(mark, p)) != std::string::npos) {
    t.erase(p, mark.size());
  }
  return t;
}

}  // namespace

std::vector<std::string> ipa_string_to_phoneme_tokens(const std::string& s) {
  std::string t = trim_copy(s);
  if (t.empty()) {
    return {};
  }
  if (t.find(' ') != std::string::npos) {
    std::vector<std::string> parts;
    std::string cur;
    for (char ch : t) {
      if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
        if (!cur.empty()) {
          parts.push_back(cur);
          cur.clear();
        }
      } else {
        cur.push_back(ch);
      }
    }
    if (!cur.empty()) {
      parts.push_back(cur);
    }
    return parts;
  }
  return utf8_split_codepoints(t);
}

int levenshtein_distance(const std::vector<std::string>& a, const std::vector<std::string>& b) {
  const int la = static_cast<int>(a.size());
  const int lb = static_cast<int>(b.size());
  if (la == 0) {
    return lb;
  }
  if (lb == 0) {
    return la;
  }
  std::vector<int> prev(static_cast<size_t>(lb + 1));
  std::vector<int> cur(static_cast<size_t>(lb + 1));
  for (int j = 0; j <= lb; ++j) {
    prev[static_cast<size_t>(j)] = j;
  }
  for (int i = 1; i <= la; ++i) {
    cur[0] = i;
    for (int j = 1; j <= lb; ++j) {
      const int cost = a[static_cast<size_t>(i - 1)] == b[static_cast<size_t>(j - 1)] ? 0 : 1;
      cur[static_cast<size_t>(j)] =
          std::min({prev[static_cast<size_t>(j)] + 1, cur[static_cast<size_t>(j - 1)] + 1,
                    prev[static_cast<size_t>(j - 1)] + cost});
    }
    prev.swap(cur);
  }
  return prev[static_cast<size_t>(lb)];
}

int pick_closest_alternative_index(const std::vector<std::string>& predicted_phoneme_tokens,
                                    const std::vector<std::string>& ipa_alternatives,
                                    const int n_valid,
                                    const int extra_phonemes) {
  const int n = std::min(n_valid, static_cast<int>(ipa_alternatives.size()));
  if (n <= 0) {
    return 0;
  }
  int best_i = 0;
  int best_d = 1'000'000'000;
  for (int i = 0; i < n; ++i) {
    const auto cand = ipa_string_to_phoneme_tokens(ipa_alternatives[static_cast<size_t>(i)]);
    const int lim = static_cast<int>(cand.size()) + std::max(0, extra_phonemes);
    std::vector<std::string> prefix;
    const int take = std::min(lim, static_cast<int>(predicted_phoneme_tokens.size()));
    prefix.assign(predicted_phoneme_tokens.begin(),
                  predicted_phoneme_tokens.begin() + static_cast<ptrdiff_t>(take));
    const int d = levenshtein_distance(cand, prefix);
    if (d < best_d) {
      best_d = d;
      best_i = i;
    }
  }
  return best_i;
}

std::string pick_closest_cmudict_ipa(const std::vector<std::string>& predicted_phoneme_tokens,
                                     const std::vector<std::string>& cmudict_alternatives,
                                     const int extra_phonemes) {
  if (cmudict_alternatives.empty()) {
    return "";
  }
  if (cmudict_alternatives.size() == 1) {
    return cmudict_alternatives[0];
  }
  const int i = pick_closest_alternative_index(
      predicted_phoneme_tokens, cmudict_alternatives,
      static_cast<int>(cmudict_alternatives.size()), extra_phonemes);
  return cmudict_alternatives[static_cast<size_t>(i)];
}

std::optional<std::string> match_prediction_to_cmudict_ipa(const std::string& predicted,
                                                             const std::vector<std::string>& alts) {
  const std::string e0 = trim_copy(predicted);
  for (const auto& alt : alts) {
    if (trim_copy(alt) == e0) {
      return alt;
    }
  }
  const std::string e1 = strip_length_markers_copy(e0);
  for (const auto& alt : alts) {
    if (strip_length_markers_copy(trim_copy(alt)) == e1) {
      return alt;
    }
  }
  return std::nullopt;
}

}  // namespace moonshine_g2p
