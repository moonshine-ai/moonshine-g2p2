#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonshine_g2p {

struct G2pWordLog;

/// Rule- and lexicon-based Russian G2P, mirroring ``russian_rule_g2p.py``.
class RussianRuleG2p {
 public:
  struct Options {
    bool with_stress = true;
    /// When true (default), move ˈ/ˌ before the syllable nucleus (same as Python / German helper).
    bool vocoder_stress = true;
  };

  explicit RussianRuleG2p(std::filesystem::path dict_tsv);
  explicit RussianRuleG2p(std::filesystem::path dict_tsv, Options options);

  [[nodiscard]] const std::string& dialect_id() const { return dialect_id_; }

  [[nodiscard]] std::string word_to_ipa(const std::string& word) const;

  [[nodiscard]] std::string text_to_ipa(const std::string& text,
                                         std::vector<G2pWordLog>* per_word_log = nullptr) const;

 private:
  std::string dialect_id_{"ru-RU"};
  Options options_;
  std::unordered_map<std::string, std::string> lexicon_;

  [[nodiscard]] std::string lookup_or_rules(const std::string& raw_word) const;
  [[nodiscard]] std::string finalize_ipa(std::string ipa) const;
};

/// True for ``ru``, ``ru-RU``, ``russian`` (case-insensitive).
[[nodiscard]] bool dialect_resolves_to_russian_rules(std::string_view dialect_id);

/// Default lexicon: ``<model-root>/../data/ru/dict.tsv`` then ``<model-root>/ru/dict.tsv``.
[[nodiscard]] std::filesystem::path resolve_russian_dict_path(const std::filesystem::path& model_root);

}  // namespace moonshine_g2p
