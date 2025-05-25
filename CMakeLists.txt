cmake_minimum_required(VERSION 3.12)
project(diffvg VERSION 0.0.1 DESCRIPTION "Differentiable Vector Graphics")

# ------------------------------------------------------------------
# 1. Python 3.11 인터프리터 + 개발 헤더 찾기
#    Interpreter 컴포넌트를 반드시 포함해야 pybind11NewTools에서 요구하는
#    Python 실행파일 정보를 얻을 수 있습니다.
find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)

# 2. pip 설치판 pybind11 찾기
#    pip 로 설치된 위치(대개 /usr/local/lib/.../pybind11/share/cmake/pybind11)
#    를 직접 CMAKE_PREFIX_PATH 로 넘겨서 CONFIG 모드로 찾습니다.
set(pybind11_DIR
    "${CMAKE_PREFIX_PATH}"
    CACHE PATH "Location of pybind11Config.cmake")
find_package(pybind11 CONFIG REQUIRED)

# 3. CUDA 옵션
option(DIFFVG_CUDA "Build diffvg with GPU code path?" ON)
if(DIFFVG_CUDA)
    message(STATUS "Build with CUDA support (modern CUDAToolkit)")
    # legacy FindCUDA 대신, CMake 3.17+ 내장 모듈인 FindCUDAToolkit 사용
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
  add_compile_definitions(COMPILE_WITH_CUDA)
  set(CMAKE_CUDA_STANDARD 11)
else()
  message(STATUS "Build without CUDA support")
  find_package(Thrust REQUIRED)
endif()

# ------------------------------------------------------------------
# 4. Include 디렉토리
#    Python_INCLUDE_DIRS 에 Python.h, frameobject.h 가 들어 있고,
#    pybind11_INCLUDE_DIRS 에 pybind11 헤더가 들어 있습니다.
include_directories(
  ${Python_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
)

# ------------------------------------------------------------------
# 5. 컴파일 옵션
if(NOT MSVC)
  add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)
endif()

# ------------------------------------------------------------------
# 6. 소스 파일 목록
set(SRCS
  atomic.h
  color.h
  cdf.h
  cuda_utils.h
  diffvg.h
  edge_query.h
  filter.h
  matrix.h
  parallel.h
  pcg.h
  ptr.h
  sample_boundary.h
  scene.h
  shape.h
  solve.h
  vector.h
  within_distance.h
  winding_number.h
  atomic.cpp
  color.cpp
  diffvg.cpp    # ← 여기 패치 필요
  parallel.cpp
  scene.cpp     # ← 여기 패치 필요
  shape.cpp
)

# ------------------------------------------------------------------
# 7. 라이브러리 생성
if(DIFFVG_CUDA)
  cuda_add_library(diffvg MODULE ${SRCS})
else()
  add_library(diffvg MODULE ${SRCS})
endif()

# ------------------------------------------------------------------
# 8. 링크 설정
#    CUDA 모드면 CUDA_LIBRARIES + Python 링크
#    non-CUDA 모드면 Thrust + Python 링크
if(DIFFVG_CUDA)
  target_link_libraries(diffvg
    PUBLIC
      CUDAToolkit::cudart   # CUDAToolkit에서 가져온 CUDA 런타임
      pybind11::pybind11
  )
else()
  target_link_libraries(diffvg
    PUBLIC
      ${THRUST_INCLUDE_DIR}
      pybind11::pybind11
  )
endif()

# ------------------------------------------------------------------
# 9. RPATH 설정 (Linux/macOS)
if(UNIX AND NOT APPLE)
  set_target_properties(diffvg PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
  )
elseif(APPLE)
  set_target_properties(diffvg PROPERTIES
    BUILD_WITH_INSTALL_RPATH "@loader_path"
  )
endif()

# ------------------------------------------------------------------
# 10. TensorFlow 바인딩은 건너뜀 (Kaggle에선 TF 빌드 실패 잦음)
#    원래 pydiffvg_tensorflow 서브디렉토리를 서브모듈로 관리하지만
#    여기선 PyTorch 전용으로만 빌드합니다.
message(STATUS "Building without TensorFlow support")
