#include "moonshine-g2p/moonshine-tts.h"

#include "moonshine-g2p/moonshine-g2p.h"
#include "moonshine-g2p/utf8-utils.h"

#include <onnxruntime_cxx_api.h>

#include <nlohmann/json.hpp>

extern "C" {
#include <utf8proc.h>
}

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace moonshine_g2p {

std::filesystem::path builtin_kokoro_bundle_dir() {
  return std::filesystem::path(__FILE__).parent_path() / "data" / "kokoro";
}

namespace {

constexpr std::string_view kVoiceMagic = "KVO1";

std::string utf8_nfc(std::string_view s) {
  const std::string tmp(s);
  utf8proc_uint8_t* p =
      utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(tmp.c_str()));
  if (p == nullptr) {
    return std::string(s);
  }
  std::string out(reinterpret_cast<char*>(p));
  std::free(p);
  return out;
}

void replace_utf8(std::string& s, std::string_view old_s, std::string_view new_s) {
  size_t pos = 0;
  while ((pos = s.find(old_s, pos)) != std::string::npos) {
    s.replace(pos, old_s.size(), new_s);
    pos += new_s.size();
  }
}

bool py_isspace_utf8_ch(std::string_view ch) {
  if (ch.empty()) {
    return false;
  }
  std::string tmp(ch);
  char32_t cp = 0;
  size_t adv = 0;
  if (!utf8_decode_at(tmp, 0, cp, adv) || adv != tmp.size()) {
    return false;
  }
  if (cp < 128) {
    return std::isspace(static_cast<unsigned char>(cp)) != 0;
  }
  const auto cat = utf8proc_category(static_cast<utf8proc_int32_t>(cp));
  return cat == UTF8PROC_CATEGORY_ZS || cat == UTF8PROC_CATEGORY_ZL || cat == UTF8PROC_CATEGORY_ZP;
}

std::string collapse_whitespace_join_single_space(const std::string& s) {
  std::u32string u = utf8_str_to_u32(s);
  std::string out;
  bool pending_space = false;
  for (char32_t cp : u) {
    const bool sp = (cp < 128 && std::isspace(static_cast<unsigned char>(cp)) != 0) ||
                    utf8proc_category(static_cast<utf8proc_int32_t>(cp)) == UTF8PROC_CATEGORY_ZS ||
                    utf8proc_category(static_cast<utf8proc_int32_t>(cp)) == UTF8PROC_CATEGORY_ZL ||
                    utf8proc_category(static_cast<utf8proc_int32_t>(cp)) == UTF8PROC_CATEGORY_ZP;
    if (sp) {
      if (!out.empty()) {
        pending_space = true;
      }
      continue;
    }
    if (pending_space) {
      utf8_append_codepoint(out, U' ');
      pending_space = false;
    }
    utf8_append_codepoint(out, cp);
  }
  return out;
}

std::string normalize_lang_key(std::string_view raw) {
  std::string s = trim_ascii_ws_copy(raw);
  for (char& c : s) {
    if (c == ' ') {
      c = '_';
    } else if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

struct LangProfile {
  char kokoro_lang = 'a';
  const char* default_voice = "af_heart";
  /// MoonshineG2P dialect id; nullptr when resolved only via ``resolve_lang_for_tts`` Spanish fallback.
  const char* g2p_dialect = "en_us";
};

const LangProfile* lookup_lang_profile(std::string_view key) {
  static const std::unordered_map<std::string, LangProfile> m{
      {"en_us", {'a', "af_heart", "en_us"}},
      {"en-us", {'a', "af_heart", "en_us"}},
      {"en", {'a', "af_heart", "en_us"}},
      {"en_gb", {'b', "bf_emma", "en_gb"}},
      {"en-gb", {'b', "bf_emma", "en_gb"}},
      // Spanish G2P must be a concrete dialect id (same default as spanish_rule_g2p.text_to_ipa).
      {"es", {'e', "ef_dora", "es-MX"}},
      {"fr", {'f', "ff_siwis", "fr"}},
      {"hi", {'h', "hf_alpha", "hi"}},
      {"it", {'i', "if_sara", "it"}},
      {"pt_br", {'p', "pf_dora", "pt_br"}},
      {"pt-br", {'p', "pf_dora", "pt_br"}},
      {"pt", {'p', "pf_dora", "pt_br"}},
      {"ja", {'j', "jf_alpha", "ja"}},
      {"jp", {'j', "jf_alpha", "ja"}},
      {"zh", {'z', "zf_xiaobei", "zh"}},
      {"zh_hans", {'z', "zf_xiaobei", "zh"}},
  };
  const std::string k(key);
  const auto it = m.find(k);
  if (it == m.end()) {
    return nullptr;
  }
  return &it->second;
}

/// Fills *profile* and *g2p_dialect* for ``MoonshineG2P`` (Kokoro locale + rule-based tag).
void resolve_lang_for_tts(const std::string& lk, const MoonshineG2POptions& opt, LangProfile& profile,
                          std::string& g2p_dialect) {
  if (const LangProfile* p = lookup_lang_profile(lk)) {
    profile = *p;
    g2p_dialect = p->g2p_dialect;
    return;
  }
  const std::string norm = normalize_rule_based_dialect_cli_key(lk);
  if (!norm.empty() && dialect_resolves_to_spanish_rules(norm, opt.spanish_narrow_obstruents)) {
    profile = {'e', "ef_dora", nullptr};
    g2p_dialect = norm;
    return;
  }
  throw std::runtime_error("MoonshineTTS: unsupported --lang key \"" + lk + "\"");
}

bool voice_prefix_ok(char kokoro_lang, std::string_view voice) {
  static const std::unordered_map<char, std::vector<std::string_view>> pref{
      {'a', {"af_", "am_"}}, {'b', {"bf_", "bm_"}}, {'e', {"ef_", "em_"}}, {'f', {"ff_"}},
      {'h', {"hf_", "hm_"}}, {'i', {"if_", "im_"}}, {'p', {"pf_", "pm_"}},
      {'j', {"jf_", "jm_"}}, {'z', {"zf_", "zm_"}},
  };
  const auto it = pref.find(kokoro_lang);
  if (it == pref.end()) {
    return true;
  }
  for (std::string_view p : it->second) {
    if (voice.size() >= p.size() && voice.substr(0, p.size()) == p) {
      return true;
    }
  }
  return false;
}

std::string select_voice_id(char kokoro_lang, std::string_view requested, std::string_view default_voice,
                            const std::filesystem::path& voices_dir) {
  std::string v = requested.empty() ? std::string(default_voice) : std::string(requested);
  if (!voice_prefix_ok(kokoro_lang, v)) {
    v = std::string(default_voice);
  }
  const auto f = voices_dir / (v + ".kokorovoice");
  if (!std::filesystem::is_regular_file(f)) {
    v = std::string(default_voice);
  }
  return v;
}

void apply_diphthong_map(std::string& s, char kokoro_lang) {
  static const std::array<std::pair<const char*, const char*>, 12> kAll{{
      {"t\u0361\u0283", "\u0287"},  // t͡ʃ → ʧ
      {"d\u0361\u0292", "\u02A4"},  // d͡ʒ → ʤ (U+02A4)
      {"t\u0283", "\u0287"},
      {"d\u0292", "\u02A4"},
      {"e\u026a", "A"},
      {"a\u026a", "I"},
      {"a\u028a", "W"},
      {"o\u028a", "O"},
      {"ə\u028a", "Q"},
      {"ɔ\u026a", "Y"},
      {"ɝ", "ɜɹ"},
      {"ɚ", "əɹ"},
  }};
  if (kokoro_lang == 'a' || kokoro_lang == 'b') {
    for (const auto& pr : kAll) {
      replace_utf8(s, pr.first, pr.second);
    }
  } else {
    for (const auto& pr : kAll) {
      if (std::strcmp(pr.first, "ɝ") == 0 || std::strcmp(pr.first, "ɚ") == 0) {
        continue;
      }
      replace_utf8(s, pr.first, pr.second);
    }
  }
}

std::string normalize_ipa_to_kokoro(std::string ipa, char kokoro_lang,
                                    const std::unordered_set<std::string>& vocab_keys) {
  ipa = utf8_nfc(trim_ascii_ws_copy(ipa));
  apply_diphthong_map(ipa, kokoro_lang);
  if (kokoro_lang == 'h') {
    replace_utf8(ipa, ".", "");
    replace_utf8(ipa, "t\u032a", "t");  // t̪
    replace_utf8(ipa, "d\u032a", "d");  // d̪
  }
  std::string kept;
  for (const std::string& ch : utf8_split_codepoints(ipa)) {
    if (vocab_keys.count(ch) != 0 || py_isspace_utf8_ch(ch)) {
      kept += ch;
    }
  }
  return collapse_whitespace_join_single_space(kept);
}

std::vector<std::string> chunk_phonemes(const std::string& ps, int max_cp = 510) {
  std::vector<std::string> chunks;
  if (ps.empty()) {
    return chunks;
  }
  const std::u32string u = utf8_str_to_u32(ps);
  if (u.size() <= static_cast<size_t>(max_cp)) {
    chunks.push_back(trim_ascii_ws_copy(ps));
    return chunks;
  }
  std::u32string rest = u;
  auto u32_to_utf8 = [](const std::u32string& x) {
    std::string o;
    for (char32_t c : x) {
      utf8_append_codepoint(o, c);
    }
    return o;
  };
  auto trim_u32 = [&u32_to_utf8](std::u32string x) {
    while (!x.empty() && x.front() == U' ') {
      x.erase(x.begin());
    }
    while (!x.empty() && x.back() == U' ') {
      x.pop_back();
    }
    return trim_ascii_ws_copy(u32_to_utf8(x));
  };
  while (!rest.empty()) {
    if (rest.size() <= static_cast<size_t>(max_cp)) {
      const std::string piece = trim_u32(rest);
      if (!piece.empty()) {
        chunks.push_back(piece);
      }
      break;
    }
    const size_t win_len = static_cast<size_t>(max_cp) + 1;
    std::u32string window = rest.substr(0, win_len);
    int cut = -1;
    for (int i = static_cast<int>(window.size()) - 1; i >= 0; --i) {
      if (window[static_cast<size_t>(i)] == U' ') {
        cut = i;
        break;
      }
    }
    if (cut <= 0) {
      cut = max_cp;
    }
    std::u32string piece32 = rest.substr(0, static_cast<size_t>(cut));
    rest = rest.substr(static_cast<size_t>(cut));
    while (!rest.empty() && rest.front() == U' ') {
      rest.erase(rest.begin());
    }
    const std::string piece = trim_u32(piece32);
    if (!piece.empty()) {
      chunks.push_back(piece);
    }
  }
  chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
                              [](const std::string& c) { return c.empty(); }),
               chunks.end());
  return chunks;
}

std::vector<int64_t> phoneme_str_to_input_ids(const std::string& phonemes,
                                              const std::unordered_map<std::string, int>& vocab) {
  std::vector<int64_t> ids;
  ids.push_back(0);
  for (const std::string& ch : utf8_split_codepoints(phonemes)) {
    const auto it = vocab.find(ch);
    if (it != vocab.end()) {
      ids.push_back(it->second);
    }
  }
  ids.push_back(0);
  return ids;
}

void read_kokorovoice(const std::filesystem::path& path, std::vector<float>& out_flat, uint32_t& rows,
                      uint32_t& cols) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("MoonshineTTS: cannot open voice file " + path.string());
  }
  char magic[4]{};
  f.read(magic, 4);
  if (!f || std::string_view(magic, 4) != kVoiceMagic) {
    throw std::runtime_error("MoonshineTTS: bad magic in " + path.string() + " (expected KVO1)");
  }
  uint32_t r = 0;
  uint32_t c = 0;
  f.read(reinterpret_cast<char*>(&r), 4);
  f.read(reinterpret_cast<char*>(&c), 4);
  if (!f || r == 0 || c == 0) {
    throw std::runtime_error("MoonshineTTS: invalid voice header in " + path.string());
  }
  const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
  out_flat.resize(n);
  f.read(reinterpret_cast<char*>(out_flat.data()), static_cast<std::streamsize>(n * sizeof(float)));
  if (!f || static_cast<size_t>(f.gcount()) != n * sizeof(float)) {
    throw std::runtime_error("MoonshineTTS: truncated voice data in " + path.string());
  }
  rows = r;
  cols = c;
}

Ort::SessionOptions make_ort_options(const std::vector<std::string>& names) {
  (void)names;
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  opts.SetIntraOpNumThreads(0);
  opts.SetInterOpNumThreads(0);
  return opts;
}

}  // namespace

struct MoonshineTTS::Impl {
  std::filesystem::path kokoro_dir_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "moonshine_tts"};
  Ort::Session session_{nullptr};
  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  std::unordered_map<std::string, int> vocab_{};
  std::unordered_set<std::string> vocab_keys_{};
  std::vector<float> voice_{};
  uint32_t voice_rows_ = 0;
  uint32_t voice_cols_ = 0;

  std::string voice_id_{};
  double speed_ = 1.0;
  char kokoro_lang_ = 'a';
  LangProfile profile_{};
  std::string g2p_dialect_{};
  MoonshineG2POptions g2p_opt_{};
  std::unique_ptr<MoonshineG2P> g2p_{};

  explicit Impl(MoonshineTTSOptions opt) : speed_(opt.speed) {
    kokoro_dir_ = opt.kokoro_dir.empty() ? builtin_kokoro_bundle_dir() : std::move(opt.kokoro_dir);
    if (!std::filesystem::is_directory(kokoro_dir_)) {
      throw std::runtime_error("MoonshineTTS: kokoro_dir is not a directory: " + kokoro_dir_.string());
    }
    const auto cfg_path = kokoro_dir_ / "config.json";
    const auto onnx_path = kokoro_dir_ / "model.onnx";
    if (!std::filesystem::is_regular_file(cfg_path)) {
      throw std::runtime_error("MoonshineTTS: missing " + cfg_path.string());
    }
    if (!std::filesystem::is_regular_file(onnx_path)) {
      throw std::runtime_error("MoonshineTTS: missing " + onnx_path.string());
    }
    {
      std::ifstream jf(cfg_path);
      nlohmann::json j;
      jf >> j;
      if (!j.contains("vocab") || !j["vocab"].is_object()) {
        throw std::runtime_error("MoonshineTTS: config.json missing vocab object");
      }
      for (auto it = j["vocab"].begin(); it != j["vocab"].end(); ++it) {
        const std::string key = it.key();
        vocab_[key] = it.value().get<int>();
        vocab_keys_.insert(key);
      }
    }
    Ort::SessionOptions session_opts = make_ort_options(opt.ort_provider_names);
#ifdef _WIN32
    const std::wstring wmodel = onnx_path.wstring();
    session_ = Ort::Session(env_, wmodel.c_str(), session_opts);
#else
    const std::string u8 = onnx_path.string();
    session_ = Ort::Session(env_, u8.c_str(), session_opts);
#endif
    const std::string lk = normalize_lang_key(opt.lang);
    g2p_opt_ = std::move(opt.g2p_options);
    if (opt.use_bundled_cpp_g2p_data) {
      g2p_opt_.model_root = builtin_cpp_data_root();
    }
    resolve_lang_for_tts(lk, g2p_opt_, profile_, g2p_dialect_);
    kokoro_lang_ = profile_.kokoro_lang;
    g2p_ = std::make_unique<MoonshineG2P>(g2p_dialect_, g2p_opt_);
    const std::filesystem::path voices_dir = kokoro_dir_ / "voices";
    voice_id_ = select_voice_id(kokoro_lang_, opt.voice, profile_.default_voice, voices_dir);
    reload_voice_tensor();
  }

  void reload_voice_tensor() {
    const auto path = kokoro_dir_ / "voices" / (voice_id_ + ".kokorovoice");
    if (!std::filesystem::is_regular_file(path)) {
      const auto pt = kokoro_dir_ / "voices" / (voice_id_ + ".pt");
      std::ostringstream msg;
      msg << "MoonshineTTS: missing voice file " << path.string();
      if (std::filesystem::is_regular_file(pt)) {
        msg << "\n  Export from PyTorch voice pack:\n  python scripts/export_kokoro_voice_for_cpp.py \""
            << pt.string() << "\" \"" << path.string() << '"';
      } else {
        msg << "\n  Install voices under " << (kokoro_dir_ / "voices").string()
            << " (e.g. python scripts/download_kokoro_onnx.py --out-dir " << kokoro_dir_.string()
            << " --voices " << voice_id_
            << "), then export:\n  python scripts/export_kokoro_voice_for_cpp.py \""
            << (kokoro_dir_ / "voices" / (voice_id_ + ".pt")).string() << "\" \"" << path.string() << '"';
      }
      throw std::runtime_error(msg.str());
    }
    read_kokorovoice(path, voice_, voice_rows_, voice_cols_);
  }

  void set_voice(std::string_view voice_id) {
    const std::filesystem::path voices_dir = kokoro_dir_ / "voices";
    voice_id_ = select_voice_id(kokoro_lang_, voice_id, profile_.default_voice, voices_dir);
    reload_voice_tensor();
  }

  void set_speed(double s) {
    if (!(s > 0.0) || !std::isfinite(s)) {
      throw std::runtime_error("MoonshineTTS: speed must be a positive finite number");
    }
    speed_ = s;
  }

  void set_lang(std::string_view lang_cli) {
    const std::string lk = normalize_lang_key(lang_cli);
    resolve_lang_for_tts(lk, g2p_opt_, profile_, g2p_dialect_);
    kokoro_lang_ = profile_.kokoro_lang;
    g2p_ = std::make_unique<MoonshineG2P>(g2p_dialect_, g2p_opt_);
    const std::filesystem::path voices_dir = kokoro_dir_ / "voices";
    voice_id_ = select_voice_id(kokoro_lang_, voice_id_, profile_.default_voice, voices_dir);
    reload_voice_tensor();
  }

  std::vector<float> synthesize(std::string_view text) {
    const std::string ipa = g2p_->text_to_ipa(text, nullptr);
    if (trim_ascii_ws_copy(ipa).empty()) {
      throw std::runtime_error("MoonshineTTS: G2P returned empty IPA");
    }
    std::string phonemes = normalize_ipa_to_kokoro(ipa, kokoro_lang_, vocab_keys_);
    if (phonemes.empty()) {
      throw std::runtime_error("MoonshineTTS: empty phoneme string after Kokoro vocabulary filter");
    }
    const std::vector<std::string> chunks = chunk_phonemes(phonemes);
    if (chunks.empty()) {
      throw std::runtime_error("MoonshineTTS: no phoneme chunks");
    }
    std::vector<float> wave_all;
    wave_all.reserve(chunks.size() * 8192);

    static const char* in_names[] = {"input_ids", "ref_s", "speed"};
    static const char* out_names[] = {"waveform"};

    for (const std::string& piece : chunks) {
      if (trim_ascii_ws_copy(piece).empty()) {
        continue;
      }
      std::vector<int64_t> ids = phoneme_str_to_input_ids(piece, vocab_);
      if (ids.size() > 512) {
        throw std::runtime_error("MoonshineTTS: phoneme token sequence too long for Kokoro (>512)");
      }
      const int64_t ntok = static_cast<int64_t>(ids.size());
      const std::array<int64_t, 2> shape_ids{1, ntok};

      const std::u32string pu = utf8_str_to_u32(piece);
      const size_t ncp = std::max<size_t>(pu.size(), 1);
      const size_t idx =
          std::min(ncp - 1, static_cast<size_t>(voice_rows_ > 0 ? voice_rows_ - 1 : 0));
      const size_t off = idx * static_cast<size_t>(voice_cols_);
      std::vector<float> ref_row(voice_cols_);
      if (off + voice_cols_ > voice_.size()) {
        throw std::runtime_error("MoonshineTTS: voice tensor index out of range");
      }
      std::copy(voice_.begin() + static_cast<std::ptrdiff_t>(off),
                voice_.begin() + static_cast<std::ptrdiff_t>(off + voice_cols_), ref_row.begin());
      const std::array<int64_t, 2> shape_ref{1, static_cast<int64_t>(voice_cols_)};

      double speed_val = speed_;
      std::vector<Ort::Value> inputs;
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          mem_, ids.data(), ids.size(), shape_ids.data(), shape_ids.size()));
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_, ref_row.data(), ref_row.size(), shape_ref.data(), shape_ref.size()));
      inputs.push_back(
          Ort::Value::CreateTensor<double>(mem_, &speed_val, 1, nullptr, 0));

      Ort::RunOptions run_opts{nullptr};
      auto outputs = session_.Run(run_opts, in_names, inputs.data(), inputs.size(), out_names, 1);
      const Ort::Value& wav = outputs[0];
      const auto ti = wav.GetTensorTypeAndShapeInfo();
      const size_t n_el = ti.GetElementCount();
      if (ti.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("MoonshineTTS: ONNX output is not float32");
      }
      const float* wptr = wav.GetTensorData<float>();
      for (size_t i = 0; i < n_el; ++i) {
        wave_all.push_back(wptr[i]);
      }
    }
    return wave_all;
  }
};

MoonshineTTS::MoonshineTTS(const MoonshineTTSOptions& opt) : impl_(std::make_unique<Impl>(opt)) {}

MoonshineTTS::~MoonshineTTS() = default;

MoonshineTTS::MoonshineTTS(MoonshineTTS&&) noexcept = default;
MoonshineTTS& MoonshineTTS::operator=(MoonshineTTS&&) noexcept = default;

void MoonshineTTS::set_voice(std::string_view voice_id) { impl_->set_voice(voice_id); }

void MoonshineTTS::set_speed(double speed) { impl_->set_speed(speed); }

void MoonshineTTS::set_lang(std::string_view lang_cli) { impl_->set_lang(lang_cli); }

std::vector<float> MoonshineTTS::synthesize(std::string_view text) { return impl_->synthesize(text); }

void write_wav_mono_pcm16(const std::filesystem::path& path, const std::vector<float>& samples) {
  // parent_path() is empty for plain filenames like "out.wav"; create_directories("") throws on some libstdc++.
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::vector<int16_t> pcm(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    float x = samples[i];
    if (!std::isfinite(x)) {
      x = 0.f;
    }
    x = std::max(-1.f, std::min(1.f, x));
    pcm[i] = static_cast<int16_t>(std::lrint(x * 32767.f));
  }
  const uint32_t sample_rate = static_cast<uint32_t>(MoonshineTTS::kSampleRateHz);
  const uint32_t num_samples = static_cast<uint32_t>(pcm.size());
  const uint32_t byte_rate = sample_rate * 2;
  const uint16_t block_align = 2;
  const uint32_t data_bytes = num_samples * 2;
  const uint32_t riff_chunk_size = 36 + data_bytes;

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("write_wav_mono_pcm16: cannot open " + path.string());
  }
  auto w4 = [&out](const char* s) { out.write(s, 4); };
  auto u32 = [&out](uint32_t v) {
    char b[4];
    b[0] = static_cast<char>(v & 0xff);
    b[1] = static_cast<char>((v >> 8) & 0xff);
    b[2] = static_cast<char>((v >> 16) & 0xff);
    b[3] = static_cast<char>((v >> 24) & 0xff);
    out.write(b, 4);
  };
  auto u16 = [&out](uint16_t v) {
    char b[2];
    b[0] = static_cast<char>(v & 0xff);
    b[1] = static_cast<char>((v >> 8) & 0xff);
    out.write(b, 2);
  };

  w4("RIFF");
  u32(riff_chunk_size);
  w4("WAVE");
  w4("fmt ");
  u32(16);
  u16(1);
  u16(1);
  u32(sample_rate);
  u32(byte_rate);
  u16(block_align);
  u16(16);
  w4("data");
  u32(data_bytes);
  out.write(reinterpret_cast<const char*>(pcm.data()),
            static_cast<std::streamsize>(pcm.size() * sizeof(int16_t)));
}

}  // namespace moonshine_g2p
