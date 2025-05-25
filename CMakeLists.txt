cmake_minimum_required(VERSION 3.17)
project(diffvg VERSION 0.0.1 DESCRIPTION "Differentiable Vector Graphics")

# ------------------------------------------------------------------
# 1) Python 3.11 인터프리터 + 개발 헤더
find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)

# ------------------------------------------------------------------
# 2) pip 설치판 헤더-온리 pybind11
set(pybind11_DIR "${CMAKE_PREFIX_PATH}"
    CACHE PATH "Location of pybind11Config.cmake")
find_package(pybind11 CONFIG REQUIRED)

# ------------------------------------------------------------------
# 3) CUDA 옵션 (modern FindCUDAToolkit 사용)
option(DIFFVG_CUDA "Build with GPU code path?" ON)
if(DIFFVG_CUDA)
    message(STATUS "Build with CUDA support (modern CUDAToolkit)")
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    add_compile_definitions(COMPILE_WITH_CUDA)
    set(CMAKE_CUDA_STANDARD 11)
else()
    message(STATUS "Build without CUDA support (CPU/Thrust mode)")
    find_package(Thrust REQUIRED)
endif()

# ------------------------------------------------------------------
# 4) Include 디렉토리
include_directories(
    ${Python_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)
if(NOT DIFFVG_CUDA)
    include_directories(${THRUST_INCLUDE_DIR})
endif()

# ------------------------------------------------------------------
# 5) 컴파일 옵션
if(NOT MSVC)
    add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)
endif()

# ------------------------------------------------------------------
# 6) 소스 파일
set(SRCS
    atomic.h         color.h         cdf.h
    cuda_utils.h     diffvg.h        edge_query.h
    filter.h         matrix.h        parallel.h
    pcg.h            ptr.h           sample_boundary.h
    scene.h          shape.h         solve.h
    vector.h         within_distance.h winding_number.h
    atomic.cpp       color.cpp       diffvg.cpp
    parallel.cpp     scene.cpp       shape.cpp
)

# ------------------------------------------------------------------
# 7) 모듈 라이브러리 생성
#    enable_language(CUDA) 덕분에 .cpp/.cu 모두 처리됨
add_library(diffvg MODULE ${SRCS})
if(DIFFVG_CUDA)
    # device 심볼 자동 연결, PIC 설정
    set_target_properties(diffvg PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# ------------------------------------------------------------------
# 8) 링크
if(DIFFVG_CUDA)
    target_link_libraries(diffvg
        PUBLIC
            CUDAToolkit::cudart
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
# 9) RPATH 설정 (Linux/macOS)
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
# 10) TensorFlow 바인딩 생략
message(STATUS "Skipping TensorFlow custom-ops build")
