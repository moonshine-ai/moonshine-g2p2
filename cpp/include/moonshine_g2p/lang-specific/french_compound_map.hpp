#pragma once

#include <string>
#include <unordered_map>

namespace moonshine_g2p::french_compound_map {

[[nodiscard]] const std::unordered_map<std::string, std::string>& cardinal_compound_ipa_entries();

}  // namespace moonshine_g2p::french_compound_map
