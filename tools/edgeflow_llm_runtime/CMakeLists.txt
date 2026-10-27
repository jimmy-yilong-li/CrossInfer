cmake_minimum_required(VERSION 3.20)

project(edgeflow_llm_runtime LANGUAGES C CXX OBJC OBJCXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

find_package(Threads REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(IOSURFACE_FRAMEWORK IOSurface REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(COREML_FRAMEWORK CoreML REQUIRED)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Enable native CPU instructions for an opt-in local build")
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_SCHED_MAX_COPIES 4 CACHE STRING "" FORCE)
set(GGML_VERSION_MAJOR 0)
set(GGML_VERSION_MINOR 0)
set(GGML_VERSION_PATCH 0)
set(GGML_VERSION "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}")

add_compile_definitions(
  GGML_VERSION=0x0
  GGML_COMMIT=0x0
  NDEBUG
  GGML_USE_CPU
  GGML_USE_METAL
)

set(EDGEFLOW_LLAMA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/llama_src CACHE PATH
  "llama.cpp source root for the native runtime")
set(EDGEFLOW_GGML_ROOT ${EDGEFLOW_LLAMA_ROOT}/ggml)
set(EDGEFLOW_GGML_INCLUDE_DIRS
  ${EDGEFLOW_GGML_ROOT}/include
  ${EDGEFLOW_GGML_ROOT}/src
  ${EDGEFLOW_GGML_ROOT}/src/include
  ${EDGEFLOW_GGML_ROOT}/src/ggml-cpu
  ${EDGEFLOW_GGML_ROOT}/src/ggml-cpu/amx
)

add_subdirectory(${EDGEFLOW_GGML_ROOT}/src ${CMAKE_CURRENT_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
set_property(TARGET ggml PROPERTY EXCLUDE_FROM_ALL TRUE)

get_target_property(EDGEFLOW_CPU_VARIANTS ggml-cpu MANUALLY_ADDED_DEPENDENCIES)
if(NOT EDGEFLOW_CPU_VARIANTS)
  set(EDGEFLOW_CPU_VARIANTS "ggml-cpu")
endif()

target_include_directories(ggml-base PRIVATE ${EDGEFLOW_GGML_INCLUDE_DIRS})
foreach(variant ${EDGEFLOW_CPU_VARIANTS})
  if(TARGET ${variant})
    target_include_directories(${variant} PRIVATE ${EDGEFLOW_GGML_INCLUDE_DIRS})
  endif()
endforeach()

file(GLOB EDGEFLOW_LLAMA_VENDOR_SOURCES CONFIGURE_DEPENDS
  ${EDGEFLOW_LLAMA_ROOT}/src/*.cpp
  ${EDGEFLOW_LLAMA_ROOT}/src/models/*.cpp
)

add_library(edgeflow_llama_vendor STATIC ${EDGEFLOW_LLAMA_VENDOR_SOURCES})
target_include_directories(edgeflow_llama_vendor
  PRIVATE
    ${EDGEFLOW_LLAMA_ROOT}/include
    ${EDGEFLOW_LLAMA_ROOT}/src
    ${EDGEFLOW_LLAMA_ROOT}/vendor
    ${EDGEFLOW_GGML_INCLUDE_DIRS}
)
target_link_libraries(edgeflow_llama_vendor
  PRIVATE
    ggml::ggml
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

set(EDGEFLOW_ANE_BRIDGE_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../crane/src/ane_bridge.m
)

set_source_files_properties(
  ${EDGEFLOW_ANE_BRIDGE_SOURCE}
  PROPERTIES
    COMPILE_FLAGS "-fvisibility=hidden"
)

add_library(edgeflow_ane_bridge_vendor STATIC
  ${EDGEFLOW_ANE_BRIDGE_SOURCE}
)
target_compile_options(edgeflow_ane_bridge_vendor PRIVATE
  -fobjc-arc
  -fobjc-exceptions
  -Wno-deprecated-declarations
)
target_include_directories(edgeflow_ane_bridge_vendor
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../crane/src
)
target_link_libraries(edgeflow_ane_bridge_vendor
  PRIVATE
    ${FOUNDATION_FRAMEWORK}
    ${IOSURFACE_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${CMAKE_DL_LIBS}
)

add_library(edgeflow_coreml_surface SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/mac_fastvlm_coreml_native.mm
)
target_compile_options(edgeflow_coreml_surface PRIVATE -fobjc-arc -fblocks)
target_link_libraries(edgeflow_coreml_surface PRIVATE
  edgeflow_ane_bridge_vendor
  ${COREML_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
  ${IOSURFACE_FRAMEWORK}
)

add_library(edgeflow_llm SHARED
  src/edgeflow_llm.cpp
  src/edgeflow_llm_ane_glue.mm
)

set(EDGEFLOW_RUNTIME_IDENTITY_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
  ${CMAKE_CURRENT_SOURCE_DIR}/include/edgeflow_llm/edgeflow_llm.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/edgeflow_llm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/session_logits.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen25_prefill.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/edgeflow_llm_ane_glue.mm
  ${EDGEFLOW_LLAMA_ROOT}/include/llama.h
  ${EDGEFLOW_LLAMA_ROOT}/src/llama.cpp
  ${EDGEFLOW_LLAMA_ROOT}/src/llama-model.cpp
  ${EDGEFLOW_LLAMA_ROOT}/src/llama-kv-cache.cpp
  ${EDGEFLOW_LLAMA_ROOT}/src/llama-kv-cache.h
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${EDGEFLOW_RUNTIME_IDENTITY_FILES}
)
set(EDGEFLOW_RUNTIME_IDENTITY_MATERIAL "")
foreach(identity_file IN LISTS EDGEFLOW_RUNTIME_IDENTITY_FILES)
  file(SHA256 ${identity_file} identity_sha256)
  string(APPEND EDGEFLOW_RUNTIME_IDENTITY_MATERIAL "${identity_sha256}")
endforeach()
string(SHA256 EDGEFLOW_RUNTIME_SOURCE_FINGERPRINT_SHA256
  "${EDGEFLOW_RUNTIME_IDENTITY_MATERIAL}"
)
target_compile_definitions(edgeflow_llm PRIVATE
  EDGEFLOW_LLM_RUNTIME_SOURCE_FINGERPRINT_SHA256="${EDGEFLOW_RUNTIME_SOURCE_FINGERPRINT_SHA256}"
)

target_include_directories(edgeflow_llm
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${EDGEFLOW_LLAMA_ROOT}/include
    ${EDGEFLOW_LLAMA_ROOT}/src
    ${EDGEFLOW_LLAMA_ROOT}/vendor
    ${EDGEFLOW_GGML_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../crane/src
)

target_link_libraries(edgeflow_llm
  PRIVATE
    edgeflow_llama_vendor
    edgeflow_ane_bridge_vendor
    ${FOUNDATION_FRAMEWORK}
    ${IOSURFACE_FRAMEWORK}
    ${METAL_FRAMEWORK}
)

set_target_properties(edgeflow_llm PROPERTIES OUTPUT_NAME edgeflow_llm)

# Retained FastVLM adapter: the same vendored backend and shared surface bridge.
add_library(crossinfer_fastvlm SHARED ../mac_fastvlm_ggml.cpp)
target_include_directories(crossinfer_fastvlm PRIVATE
  ${EDGEFLOW_LLAMA_ROOT}/include
  ${EDGEFLOW_GGML_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../crane/src
)
target_link_libraries(crossinfer_fastvlm PRIVATE
  edgeflow_llama_vendor edgeflow_ane_bridge_vendor ggml::ggml
  ${FOUNDATION_FRAMEWORK} ${IOSURFACE_FRAMEWORK} ${METAL_FRAMEWORK}
)
