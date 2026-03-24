#include "moonshine_g2p/moonshine_onnx_g2p.hpp"

#include "moonshine_g2p/onnx_g2p_models.hpp"
#include "moonshine_g2p/text_normalize.hpp"
#include "moonshine_g2p/utf8_utils.hpp"

#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace moonshine_g2p {

namespace {

void append_log(std::vector<G2pWordLog>* out, G2pWordLog entry) {
  if (out != nullptr) {
    out->push_back(std::move(entry));
  }
}

}  // namespace

MoonshineOnnxG2p::MoonshineOnnxG2p(CmudictTsv dict,
                                   std::optional<std::filesystem::path> heteronym_onnx,
                                   std::optional<std::filesystem::path> oov_onnx,
                                   bool use_cuda,
                                   bool require_heteronym_model,
                                   bool require_oov_model)
    : dict_(std::move(dict)) {
  if (require_heteronym_model) {
    if (!heteronym_onnx.has_value()) {
      throw std::runtime_error("heteronym ONNX model required but no path was provided");
    }
    if (!std::filesystem::is_regular_file(*heteronym_onnx)) {
      throw std::runtime_error("heteronym ONNX model required but file not found: " +
                               heteronym_onnx->generic_string());
    }
  }
  if (require_oov_model) {
    if (!oov_onnx.has_value()) {
      throw std::runtime_error("OOV ONNX model required but no path was provided");
    }
    if (!std::filesystem::is_regular_file(*oov_onnx)) {
      throw std::runtime_error("OOV ONNX model required but file not found: " +
                               oov_onnx->generic_string());
    }
  }
  if (heteronym_onnx && std::filesystem::is_regular_file(*heteronym_onnx)) {
    het_ = std::make_unique<OnnxHeteronymG2p>(env_, *heteronym_onnx, use_cuda);
  }
  if (oov_onnx && std::filesystem::is_regular_file(*oov_onnx)) {
    oov_ = std::make_unique<OnnxOovG2p>(env_, *oov_onnx, use_cuda);
  }
}

MoonshineOnnxG2p::~MoonshineOnnxG2p() = default;

std::string MoonshineOnnxG2p::text_to_ipa(std::string_view text, std::vector<G2pWordLog>* per_word_log) {
  const std::string full_text(text);
  std::vector<std::string> parts;
  int pos = 0;

  for (const auto& token : split_text_to_words(full_text)) {
    std::optional<std::pair<int, int>> se = utf8_find_token_codepoints(full_text, token, pos);
    if (!se) {
      se = utf8_find_token_codepoints(full_text, token, 0);
    }
    if (!se) {
      append_log(per_word_log,
                 G2pWordLog{token, "", G2pWordPath::kTokenNotLocatedInText, ""});
      continue;
    }
    const int start = se->first;
    const int end = se->second;
    pos = end;

    const std::string key = normalize_word_for_lookup(token);
    if (key.empty()) {
      append_log(per_word_log, G2pWordLog{token, "", G2pWordPath::kSkippedEmptyKey, ""});
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
          append_log(per_word_log,
                     G2pWordLog{token, key, G2pWordPath::kOovModel, chunk});
          parts.push_back(std::move(chunk));
        } else {
          append_log(per_word_log,
                     G2pWordLog{token, key, G2pWordPath::kOovModelNoOutput, ""});
        }
      } else {
        append_log(per_word_log,
                   G2pWordLog{token, key, G2pWordPath::kUnknownNoOovModel, ""});
      }
      continue;
    }
    const std::vector<std::string>& alts = *alts_ptr;
    if (alts.size() == 1) {
      append_log(per_word_log,
                 G2pWordLog{token, key, G2pWordPath::kDictUnambiguous, alts[0]});
      parts.push_back(alts[0]);
    } else if (het_) {
      const std::string ipa = het_->disambiguate_ipa(full_text, start, end, key, alts);
      append_log(per_word_log,
                 G2pWordLog{token, key, G2pWordPath::kDictHeteronym, ipa});
      parts.push_back(std::move(ipa));
    } else {
      append_log(per_word_log,
                 G2pWordLog{token, key, G2pWordPath::kDictFirstAlternativeNoHeteronymModel,
                            alts[0]});
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
