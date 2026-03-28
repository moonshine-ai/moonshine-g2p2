#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace moonshine_g2p::spanish_unicode {

// Definitions in spanish_unicode_tables.cpp (generated Unicode data).
extern const std::pair<char32_t, const char*> k_unicode_strip_table[];
extern const std::size_t k_unicode_strip_table_size;
extern const std::pair<char32_t, const char*> k_unicode_lower_table[];
extern const std::size_t k_unicode_lower_table_size;
extern const std::uint32_t k_unicode_word_bitmap[];
extern const std::uint32_t k_unicode_word_bitmap_words;
extern const std::uint32_t k_unicode_space_bitmap[];
extern const std::uint32_t k_unicode_space_bitmap_words;

}  // namespace moonshine_g2p::spanish_unicode
