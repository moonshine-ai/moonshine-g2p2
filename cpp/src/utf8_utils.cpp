#include "moonshine_g2p/utf8_utils.hpp"

namespace moonshine_g2p {

void utf8_append_codepoint(std::string& out, char32_t cp) {
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0x10FFFF) {
    out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::vector<std::string> utf8_split_codepoints(const std::string& utf8) {
  std::vector<std::string> out;
  const size_t n = utf8.size();
  size_t i = 0;
  while (i < n) {
    const unsigned char c0 = static_cast<unsigned char>(utf8[i]);
    char32_t cp = 0;
    size_t adv = 1;
    if (c0 < 0x80) {
      cp = c0;
    } else if ((c0 >> 5) == 0x6 && i + 1 < n) {
      const unsigned char c1 = static_cast<unsigned char>(utf8[i + 1]);
      if ((c1 >> 6) != 0x2) {
        out.emplace_back(1, utf8[i++]);
        continue;
      }
      cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
      adv = 2;
    } else if ((c0 >> 4) == 0xE && i + 2 < n) {
      const unsigned char c1 = static_cast<unsigned char>(utf8[i + 1]);
      const unsigned char c2 = static_cast<unsigned char>(utf8[i + 2]);
      if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2) {
        out.emplace_back(1, utf8[i++]);
        continue;
      }
      cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
      adv = 3;
    } else if ((c0 >> 3) == 0x1E && i + 3 < n) {
      const unsigned char c1 = static_cast<unsigned char>(utf8[i + 1]);
      const unsigned char c2 = static_cast<unsigned char>(utf8[i + 2]);
      const unsigned char c3 = static_cast<unsigned char>(utf8[i + 3]);
      if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2 || (c3 >> 6) != 0x2) {
        out.emplace_back(1, utf8[i++]);
        continue;
      }
      cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
      adv = 4;
    } else {
      out.emplace_back(1, utf8[i++]);
      continue;
    }
    std::string one;
    utf8_append_codepoint(one, cp);
    out.push_back(std::move(one));
    i += adv;
  }
  return out;
}

std::optional<std::pair<int, int>> utf8_find_token_codepoints(const std::string& text,
                                                              const std::string& token,
                                                              int start_cp) {
  const auto hay = utf8_split_codepoints(text);
  const auto nd = utf8_split_codepoints(token);
  if (nd.empty()) {
    return std::nullopt;
  }
  const int hn = static_cast<int>(hay.size());
  const int tn = static_cast<int>(nd.size());
  if (start_cp < 0) {
    start_cp = 0;
  }
  for (int i = start_cp; i + tn <= hn; ++i) {
    bool ok = true;
    for (int j = 0; j < tn; ++j) {
      if (hay[static_cast<size_t>(i + j)] != nd[static_cast<size_t>(j)]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return std::make_pair(i, i + tn);
    }
  }
  // Fallback: first occurrence (Python text.find(token))
  for (int i = 0; i + tn <= hn; ++i) {
    bool ok = true;
    for (int j = 0; j < tn; ++j) {
      if (hay[static_cast<size_t>(i + j)] != nd[static_cast<size_t>(j)]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return std::make_pair(i, i + tn);
    }
  }
  return std::nullopt;
}

}  // namespace moonshine_g2p
