#include "moonshine_g2p/moonshine_onnx_g2p.hpp"

#include "moonshine_g2p/onnx_g2p_models.hpp"
#include "moonshine_g2p/text_normalize.hpp"
#include "moonshine_g2p/utf8_utils.hpp"

#include <sstream>

namespace moonshine_g2p {

MoonshineOnnxG2p::MoonshineOnnxG2p(CmudictTsv dict,
                                   std::optional<std::filesystem::path> heteronym_onnx,
                                   std::optional<std::filesystem::path> oov_onnx,
                                   bool use_cuda)
    : dict_(std::move(dict)) {
  if (heteronym_onnx) {
    het_ = std::make_unique<OnnxHeteronymG2p>(env_, *heteronym_onnx, use_cuda);
  }
  if (oov_onnx) {
    oov_ = std::make_unique<OnnxOovG2p>(env_, *oov_onnx, use_cuda);
  }
}

MoonshineOnnxG2p::~MoonshineOnnxG2p() = default;

std::string MoonshineOnnxG2p::text_to_ipa(std::string_view text) {
  const std::string full_text(text);
  std::vector<std::string> parts;
  int pos = 0;

  for (const auto& token : split_text_to_words(full_text)) {
    std::optional<std::pair<int, int>> se = utf8_find_token_codepoints(full_text, token, pos);
    if (!se) {
      se = utf8_find_token_codepoints(full_text, token, 0);
    }
    if (!se) {
      continue;
    }
    const int start = se->first;
    const int end = se->second;
    pos = end;

    const std::string key = normalize_word_for_lookup(token);
    if (key.empty()) {
      continue;
    }

    const std::vector<std::string>* alts_ptr = dict_.lookup(key);
    if (alts_ptr == nullptr || alts_ptr->empty()) {
      if (oov_) {
        const std::vector<std::string> phones = oov_->predict_phonemes(key);
        if (!phones.empty()) {
          std::string chunk;
          for (const auto& p : phones) {
            chunk += p;
          }
          parts.push_back(std::move(chunk));
        }
      }
      continue;
    }
    const std::vector<std::string>& alts = *alts_ptr;
    if (alts.size() == 1) {
      parts.push_back(alts[0]);
    } else if (het_) {
      parts.push_back(het_->disambiguate_ipa(full_text, start, end, key, alts));
    } else {
      parts.push_back(alts[0]);
    }
  }

  std::ostringstream out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      out << ' ';
    }
    out << parts[i];
  }
  return out.str();
}

}  // namespace moonshine_g2p
