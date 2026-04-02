#ifndef MOONSHINE_G2P_MOONSHINE_TTS_H
#define MOONSHINE_G2P_MOONSHINE_TTS_H

#include "moonshine-g2p/builtin-cpp-data-root.h"
#include "moonshine-g2p/moonshine-g2p-options.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_g2p {

/// Default Kokoro bundle: ``cpp/data/kokoro`` next to ``moonshine-tts.cpp`` (``config.json``, ``model.onnx``, ``voices/``).
std::filesystem::path builtin_kokoro_bundle_dir();

/// Options for ``MoonshineTTS`` (Kokoro ONNX bundle + Moonshine G2P).
struct MoonshineTTSOptions {
  /// Directory with ``config.json``, ``model.onnx``, and ``voices/*.kokorovoice`` (see
  /// ``scripts/export_kokoro_voice_for_cpp.py``). Empty → ``builtin_kokoro_bundle_dir()``.
  std::filesystem::path kokoro_dir{};
  /// Locale tag: built-ins like ``en_us``, ``es``, ``ja``, or any Spanish rule id (``es_mx``, ``es-AR``, …).
  std::string lang = "en_us";
  /// Kokoro voice id (e.g. ``af_heart``). Empty → default for ``lang``.
  std::string voice{};
  double speed = 1.0;
  /// When true (default), ``g2p_options.model_root`` is replaced with ``builtin_cpp_data_root()`` (the
  /// ``cpp/data`` tree: ``ja/``, ``en_us/``, ``zh_hans/``, …). Set false to use your own ``model_root``.
  bool use_bundled_cpp_g2p_data = true;
  MoonshineG2POptions g2p_options{};
  /// ONNX Runtime provider names (e.g. ``CUDAExecutionProvider``). Empty → CPU only.
  std::vector<std::string> ort_provider_names{};
};

/// True when ``MoonshineTTS`` accepts ``lang_cli`` (built-in Kokoro locales plus Spanish rule dialects).
/// Used by ``moonshine_tts`` CLI to fall back to Piper for other languages.
bool kokoro_tts_lang_supported(std::string_view lang_cli, const MoonshineG2POptions& g2p_opt = {});

/// Kokoro-82M via ONNX Runtime, with IPA from ``MoonshineG2P`` (same role as ``speak.py`` ONNX path).
class MoonshineTTS {
 public:
  explicit MoonshineTTS(const MoonshineTTSOptions& opt);
  MoonshineTTS(const MoonshineTTS&) = delete;
  MoonshineTTS& operator=(const MoonshineTTS&) = delete;
  MoonshineTTS(MoonshineTTS&&) noexcept;
  MoonshineTTS& operator=(MoonshineTTS&&) noexcept;
  ~MoonshineTTS();

  void set_voice(std::string_view voice_id);
  void set_speed(double speed);
  void set_lang(std::string_view lang_cli);

  static constexpr int kSampleRateHz = 24000;

  /// Text → IPA (MoonshineG2P) → Kokoro phoneme string → ONNX → mono float waveform.
  std::vector<float> synthesize(std::string_view text);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Writes mono 16-bit PCM WAV at ``MoonshineTTS::kSampleRateHz`` (samples clipped to [-1, 1]).
void write_wav_mono_pcm16(const std::filesystem::path& path, const std::vector<float>& samples);

}  // namespace moonshine_g2p

#endif  // MOONSHINE_G2P_MOONSHINE_TTS_H
