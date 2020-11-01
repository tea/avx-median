cmake_minimum_required (VERSION 3.8)

project ("avx-median")

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(Celero)

add_executable (avx-median
	avx-median.cpp
	avx-median.h
	step0.cpp
	step1.cpp
	step2.cpp
	step3.cpp
	parallel_avx512.cpp
	parallel_avx2.cpp
	)

target_link_libraries(avx-median PRIVATE celero)

if(MSVC)
target_compile_options(avx-median PRIVATE /wd4251 /wd4700)
target_compile_options(avx-median PRIVATE /arch:AVX512)
else()
target_compile_options(avx-median PRIVATE -march=skylake-avx512)
endif()