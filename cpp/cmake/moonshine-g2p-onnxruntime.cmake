# Resolve ONNX Runtime for moonshine-g2p C++.
#
# Default (no -DONNXRUNTIME_ROOT): bundled tree under this repo:
#   cpp/third-party/onnxruntime/include/
#   cpp/third-party/onnxruntime/lib/linux/x86_64/libonnxruntime.so.1
#
# Override with -DONNXRUNTIME_ROOT=/other/path (same include/ + your lib layout).
#
# Style aligned with https://github.com/moonshine-ai/moonshine core/third-party/onnxruntime.

# This file lives in cpp/cmake/; default ORT root is cpp/third-party/onnxruntime
set(_MOONSHINE_G2P_ORT_DEFAULT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../third-party/onnxruntime")
set(ONNXRUNTIME_ROOT "${_MOONSHINE_G2P_ORT_DEFAULT_ROOT}" CACHE PATH
    "Root of ONNX Runtime distribution (default: cpp/third-party/onnxruntime)")

set(MOONSHINE_G2P_ORT_INCLUDE "${ONNXRUNTIME_ROOT}/include")

set(_ort_linux_x64 "${ONNXRUNTIME_ROOT}/lib/linux/x86_64/libonnxruntime.so.1")

if(WIN32)
  find_library(
    MOONSHINE_G2P_ORT_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib/windows/x86_64"
    NO_DEFAULT_PATH
    REQUIRED
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "${_ort_linux_x64}")
  set(MOONSHINE_G2P_ORT_LIB "${_ort_linux_x64}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_library(
    MOONSHINE_G2P_ORT_LIB
    NAMES onnxruntime libonnxruntime.so.1
    PATHS "${ONNXRUNTIME_ROOT}/lib/linux/x86_64" "${ONNXRUNTIME_ROOT}/lib/linux/aarch64"
          "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
elseif(APPLE)
  find_library(
    MOONSHINE_G2P_ORT_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib/macos/arm64" "${ONNXRUNTIME_ROOT}/lib/macos/x86_64"
          "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
else()
  find_library(
    MOONSHINE_G2P_ORT_LIB
    NAMES onnxruntime libonnxruntime.so.1
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

if(NOT EXISTS "${MOONSHINE_G2P_ORT_INCLUDE}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "onnxruntime_cxx_api.h not found under ${MOONSHINE_G2P_ORT_INCLUDE}")
endif()

function(moonshine_g2p_link_onnxruntime target_name)
  target_include_directories(${target_name} PUBLIC "${MOONSHINE_G2P_ORT_INCLUDE}")
  target_link_libraries(${target_name} PUBLIC "${MOONSHINE_G2P_ORT_LIB}")
  if(UNIX AND NOT APPLE)
    target_link_libraries(${target_name} PUBLIC pthread dl)
  endif()
endfunction()

# Link .so from a non-system path: set RPATH on runnable targets that link ORT.
function(moonshine_g2p_set_ort_rpath target_name)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND MOONSHINE_G2P_ORT_LIB MATCHES "\\.so")
    get_filename_component(_moonshine_g2p_ort_rpath "${MOONSHINE_G2P_ORT_LIB}" DIRECTORY)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${_moonshine_g2p_ort_rpath}"
      INSTALL_RPATH "${_moonshine_g2p_ort_rpath}"
    )
  endif()
endfunction()
