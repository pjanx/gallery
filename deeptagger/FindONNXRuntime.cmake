# Public Domain

find_path (ONNXRuntime_INCLUDE_DIRS onnxruntime_c_api.h
	PATH_SUFFIXES onnxruntime)
find_library (ONNXRuntime_LIBRARIES NAMES onnxruntime)

include (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (ONNXRuntime DEFAULT_MSG
	ONNXRuntime_INCLUDE_DIRS ONNXRuntime_LIBRARIES)

mark_as_advanced (ONNXRuntime_LIBRARIES ONNXRuntime_INCLUDE_DIRS)
