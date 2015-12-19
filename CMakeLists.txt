PROJECT(srba_stereo_slam)
CMAKE_MINIMUM_REQUIRED(VERSION 2.4.5)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

# Sources:
add_subdirectory(src)

