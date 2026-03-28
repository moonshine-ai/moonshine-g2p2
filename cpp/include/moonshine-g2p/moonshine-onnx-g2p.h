#ifndef MOONSHINE_G2P_MOONSHINE_ONNX_G2P_H
#define MOONSHINE_G2P_MOONSHINE_ONNX_G2P_H

#include "moonshine-g2p/cmudict-tsv.h"
#include "moonshine-g2p/g2p-word-log.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_g2p {

class OnnxHeteronymG2p;
class OnnxOovG2p;

// C++ counterpart to ``moonshine_onnx_g2p.py`` (no eSpeak).
class MoonshineOnnxG2p {
 public:
  /// *heteronym_onnx* / *oov_onnx*: pass a filesystem path when a model should be loaded (file must
  /// exist).
  MoonshineOnnxG2p(std::optional<std::filesystem::path> dict_path,
                   std::optional<std::filesystem::path> heteronym_onnx,
                   std::optional<std::filesystem::path> oov_onnx,
                   bool use_cuda = false);
  ~MoonshineOnnxG2p();

  /// Converts *text* to a space-separated IPA line. If *per_word_log* is non-null, appends one
  /// G2pWordLog per surface token (including skips and unknowns).
  std::string text_to_ipa(std::string_view text,
                                        std::vector<G2pWordLog>* per_word_log = nullptr);

 private:
  std::unique_ptr<CmudictTsv> dict_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "moonshine_g2p"};
  std::unique_ptr<OnnxHeteronymG2p> het_;
  std::unique_ptr<OnnxOovG2p> oov_;
};

}  // namespace moonshine_g2p

#endif  // MOONSHINE_G2P_MOONSHINE_ONNX_G2P_H
