# Ubuntu 20.04 LTS
cmake_minimum_required (VERSION 3.16)
project (deeptagger VERSION 0.0.1 LANGUAGES CXX)

# Hint: set ONNXRuntime_ROOT to a directory with a pre-built GitHub release.
# (Useful for development, otherwise you may need to adjust the rpath.)
set (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")

find_package (ONNXRuntime REQUIRED)
find_package (PkgConfig REQUIRED)
pkg_check_modules (GM REQUIRED GraphicsMagick++)

add_executable (deeptagger deeptagger.cpp)
target_compile_features (deeptagger PRIVATE cxx_std_17)
target_include_directories (deeptagger PRIVATE
	${GM_INCLUDE_DIRS} ${ONNXRuntime_INCLUDE_DIRS})
target_link_directories (deeptagger PRIVATE
	${GM_LIBRARY_DIRS})
target_link_libraries (deeptagger PRIVATE
	${GM_LIBRARIES} ${ONNXRuntime_LIBRARIES})
