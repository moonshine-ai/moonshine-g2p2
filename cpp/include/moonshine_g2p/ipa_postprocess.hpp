#pragma once

#include <optional>
#include <string>
#include <vector>

namespace moonshine_g2p {

std::vector<std::string> ipa_string_to_phoneme_tokens(const std::string& s);

int levenshtein_distance(const std::vector<std::string>& a, const std::vector<std::string>& b);

int pick_closest_alternative_index(const std::vector<std::string>& predicted_phoneme_tokens,
                                    const std::vector<std::string>& ipa_alternatives,
                                    int n_valid,
                                    int extra_phonemes);

std::string pick_closest_cmudict_ipa(const std::vector<std::string>& predicted_phoneme_tokens,
                                     const std::vector<std::string>& cmudict_alternatives,
                                     int extra_phonemes);

// Returns canonical string from alternatives or nullopt (Python None).
std::optional<std::string> match_prediction_to_cmudict_ipa(const std::string& predicted,
                                                             const std::vector<std::string>& alts);

}  // namespace moonshine_g2p
