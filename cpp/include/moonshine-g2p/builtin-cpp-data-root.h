#ifndef MOONSHINE_G2P_BUILTIN_CPP_DATA_ROOT_H
#define MOONSHINE_G2P_BUILTIN_CPP_DATA_ROOT_H

#include <filesystem>

namespace moonshine_g2p {

/// ``cpp/data`` next to this library’s sources (lexicons, JA/ZH/AR ONNX bundles, …).
std::filesystem::path builtin_cpp_data_root();

}  // namespace moonshine_g2p

#endif
