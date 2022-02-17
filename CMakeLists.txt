cmake_minimum_required(VERSION 3.0.0)
project(tt VERSION 0.1.0)


set(CMAKE_C_FLAGS "-Wall -pthread ")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}  -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

set(CMAKE_BUILD_TYPE Debug)

include_directories("./include")
include_directories("/usr/include/opencv4/")

set(OpenCV_LIBS opencv_highgui opencv_core opencv_imgcodecs opencv_imgproc opencv_video opencv_videoio)
set(EXTERN_LIBS ${PROJECT_SOURCE_DIR}/libs)

aux_source_directory("./src" DIR_SRCS)



link_directories(${EXTERN_LIBS})
link_directories(${EXTERN_LIBS}/pthreadpool/)
link_directories(${EXTERN_LIBS}/_deps/flatbuffers-build/)
link_directories(${EXTERN_LIBS}/_deps/ruy-build/ruy/)
link_directories(${EXTERN_LIBS}/_deps/xnnpack-build/)
link_directories(${EXTERN_LIBS}/_deps/cpuinfo-build/)
link_directories(${EXTERN_LIBS}/_deps/clog-build/)
link_directories(${EXTERN_LIBS}/_deps/fft2d-build/)
link_directories(${EXTERN_LIBS}/_deps/farmhash-build/)


add_executable(${PROJECT_NAME} ${DIR_SRCS} main.cpp)
target_link_libraries(${PROJECT_NAME} tensorflow-lite)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} flatbuffers)
target_link_libraries(${PROJECT_NAME} ruy_context_get_ctx 
ruy_context 
ruy_frontend 
ruy_prepare_packed_matrices 
ruy_pack_avx 
ruy_kernel_avx 
ruy_pack_avx2_fma
ruy_kernel_avx2_fma 
ruy_pack_avx512 
ruy_kernel_avx512 
ruy_pack_arm 
ruy_kernel_arm 
ruy_apply_multiplier
ruy_trmul
ruy_block_map
ruy_ctx
ruy_have_built_path_for_avx
ruy_have_built_path_for_avx2_fma
ruy_have_built_path_for_avx512
ruy_tune
ruy_cpuinfo
ruy_prepacked_cache
ruy_thread_pool
ruy_denormal
ruy_blocking_counter
ruy_wait
ruy_allocator
ruy_system_aligned_alloc
)
target_link_libraries(${PROJECT_NAME} XNNPACK)
target_link_libraries(${PROJECT_NAME} cpuinfo)
target_link_libraries(${PROJECT_NAME} clog)
target_link_libraries(${PROJECT_NAME} pthreadpool)
target_link_libraries(${PROJECT_NAME} fft2d_fftsg fft2d_fftsg2d )
target_link_libraries(${PROJECT_NAME} farmhash)

target_link_libraries(${PROJECT_NAME} dl)
