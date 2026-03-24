#pragma once

#include "moonshine_g2p/cmudict_tsv.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <string_view>

namespace moonshine_g2p {

class OnnxHeteronymG2p;
class OnnxOovG2p;

// C++ counterpart to ``moonshine_onnx_g2p.py`` (no eSpeak).
class MoonshineOnnxG2p {
 public:
  MoonshineOnnxG2p(CmudictTsv dict,
                   std::optional<std::filesystem::path> heteronym_onnx,
                   std::optional<std::filesystem::path> oov_onnx,
                   bool use_cuda = false);
  ~MoonshineOnnxG2p();

  [[nodiscard]] std::string text_to_ipa(std::string_view text);

 private:
  CmudictTsv dict_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "moonshine_g2p"};
  std::unique_ptr<OnnxHeteronymG2p> het_;
  std::unique_ptr<OnnxOovG2p> oov_;
};

}  // namespace moonshine_g2p
