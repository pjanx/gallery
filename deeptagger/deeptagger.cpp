#include <getopt.h>
#include <Magick++.h>
#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

#include <cstdio>
#include <cstdint>
#include <climits>

static struct {
	bool cpu = false;
	int debug = 0;
	long batch = 1;
	float threshold = 0.1;

	// Execution provider name → Key → Value
	std::map<std::string, std::map<std::string, std::string>> options;
} g;

// --- Configuration -----------------------------------------------------------

// Arguably, input normalization could be incorporated into models instead.
struct Config {
	std::string name;
	enum class Shape {NHWC, NCHW} shape = Shape::NHWC;
	enum class Channels {RGB, BGR} channels = Channels::RGB;
	bool normalize = false;
	enum class Pad {WHITE, EDGE, STRETCH} pad = Pad::WHITE;
	int size = -1;
	bool sigmoid = false;

	std::vector<std::string> tags;
};

static void
read_tags(const std::string &path, std::vector<std::string> &tags)
{
	std::ifstream f(path);
	f.exceptions(std::ifstream::badbit);
	if (!f)
		throw std::runtime_error("cannot read tags");

	std::string line;
	while (std::getline(f, line)) {
		if (!line.empty() && line.back() == '\r')
			line.erase(line.size() - 1);
		tags.push_back(line);
	}
}

static void
read_field(Config &config, std::string key, std::string value)
{
	if (key == "name") {
		config.name = value;
	} else if (key == "shape") {
		if      (value == "nhwc")    config.shape = Config::Shape::NHWC;
		else if (value == "nchw")    config.shape = Config::Shape::NCHW;
		else throw std::invalid_argument("bad value for: " + key);
	} else if (key == "channels") {
		if      (value == "rgb")     config.channels = Config::Channels::RGB;
		else if (value == "bgr")     config.channels = Config::Channels::BGR;
		else throw std::invalid_argument("bad value for: " + key);
	} else if (key == "normalize") {
		if      (value == "true")    config.normalize = true;
		else if (value == "false")   config.normalize = false;
		else throw std::invalid_argument("bad value for: " + key);
	} else if (key == "pad") {
		if      (value == "white")   config.pad = Config::Pad::WHITE;
		else if (value == "edge")    config.pad = Config::Pad::EDGE;
		else if (value == "stretch") config.pad = Config::Pad::STRETCH;
		else throw std::invalid_argument("bad value for: " + key);
	} else if (key == "size") {
		config.size = std::stoi(value);
	} else if (key == "interpret") {
		if      (value == "false")   config.sigmoid = false;
		else if (value == "sigmoid") config.sigmoid = true;
		else throw std::invalid_argument("bad value for: " + key);
	} else {
		throw std::invalid_argument("unsupported config key: " + key);
	}
}

static void
read_config(Config &config, const char *path)
{
	std::ifstream f(path);
	f.exceptions(std::ifstream::badbit);
	if (!f)
		throw std::runtime_error("cannot read configuration");

	std::regex re(R"(^\s*([^#=]+?)\s*=\s*([^#]*?)\s*(?:#|$))",
		std::regex::optimize);
	std::smatch m;

	std::string line;
	while (std::getline(f, line)) {
		if (std::regex_match(line, m, re))
			read_field(config, m[1].str(), m[2].str());
	}

	read_tags(
		std::filesystem::path(path).replace_extension("tags"), config.tags);
}

// --- Data preparation --------------------------------------------------------

static float *
image_to_nhwc(float *data, Magick::Image &image, Config::Channels channels)
{
	unsigned int width = image.columns();
	unsigned int height = image.rows();

	auto pixels = image.getConstPixels(0, 0, width, height);
	switch (channels) {
	case Config::Channels::RGB:
		for (unsigned int y = 0; y < height; y++) {
			for (unsigned int x = 0; x < width; x++) {
				auto pixel = *pixels++;
				*data++ = ScaleQuantumToChar(pixel.red);
				*data++ = ScaleQuantumToChar(pixel.green);
				*data++ = ScaleQuantumToChar(pixel.blue);
			}
		}
		break;
	case Config::Channels::BGR:
		for (unsigned int y = 0; y < height; y++) {
			for (unsigned int x = 0; x < width; x++) {
				auto pixel = *pixels++;
				*data++ = ScaleQuantumToChar(pixel.blue);
				*data++ = ScaleQuantumToChar(pixel.green);
				*data++ = ScaleQuantumToChar(pixel.red);
			}
		}
	}
	return data;
}

static float *
image_to_nchw(float *data, Magick::Image &image, Config::Channels channels)
{
	unsigned int width = image.columns();
	unsigned int height = image.rows();

	auto pixels = image.getConstPixels(0, 0, width, height), pp = pixels;
	switch (channels) {
	case Config::Channels::RGB:
		for (unsigned int y = 0; y < height; y++)
			for (unsigned int x = 0; x < width; x++)
				*data++ = ScaleQuantumToChar((*pp++).red);
		pp = pixels;
		for (unsigned int y = 0; y < height; y++)
			for (unsigned int x = 0; x < width; x++)
				*data++ = ScaleQuantumToChar((*pp++).green);
		pp = pixels;
		for (unsigned int y = 0; y < height; y++)
			for (unsigned int x = 0; x < width; x++)
				*data++ = ScaleQuantumToChar((*pp++).blue);
		break;
	case Config::Channels::BGR:
		for (unsigned int y = 0; y < height; y++)
			for (unsigned int x = 0; x < width; x++)
				*data++ = ScaleQuantumToChar((*pp++).blue);
		pp = pixels;
		for (unsigned int y = 0; y < height; y++)
			for (unsigned int x = 0; x < width; x++)
				*data++ = ScaleQuantumToChar((*pp++).green);
		pp = pixels;
		for (unsigned int y = 0; y < height; y++)
			for (unsigned int x = 0; x < width; x++)
				*data++ = ScaleQuantumToChar((*pp++).red);
	}
	return data;
}

static Magick::Image
load(const std::string filename,
	const Config &config, int64_t width, int64_t height)
{
	Magick::Image image;
	try {
		image.read(filename);
	} catch (const Magick::Warning &warning) {
		if (g.debug)
			fprintf(stderr, "%s: %s\n", filename.c_str(), warning.what());
	}

	image.autoOrient();

	Magick::Geometry adjusted(width, height);
	switch (config.pad) {
	case Config::Pad::EDGE:
	case Config::Pad::WHITE:
		adjusted.greater(true);
		break;
	case Config::Pad::STRETCH:
		adjusted.aspect(false);
	}

	image.resize(adjusted, Magick::LanczosFilter);

	// The GraphicsMagick API doesn't offer any good options.
	if (config.pad == Config::Pad::EDGE) {
		MagickLib::SetImageVirtualPixelMethod(
			image.image(), MagickLib::EdgeVirtualPixelMethod);

		auto x = (int64_t(image.columns()) - width) / 2;
		auto y = (int64_t(image.rows()) - height) / 2;
		auto source = image.getConstPixels(x, y, width, height);
		std::vector<MagickLib::PixelPacket>
			pixels(source, source + width * height);

		Magick::Image edged(Magick::Geometry(width, height), "black");
		edged.classType(Magick::DirectClass);
		auto target = edged.setPixels(0, 0, width, height);
		memcpy(target, pixels.data(), pixels.size() * sizeof pixels[0]);
		edged.syncPixels();

		image = edged;
	}

	// Center it in a square patch of white, removing any transparency.
	// image.extent() could probably be used to do the same thing.
	Magick::Image white(Magick::Geometry(width, height), "white");
	auto x = (white.columns() - image.columns()) / 2;
	auto y = (white.rows() - image.rows()) / 2;
	white.composite(image, x, y, Magick::OverCompositeOp);
	white.fileName(filename);

	if (g.debug > 2)
		white.display();

	return white;
}

// --- Inference ---------------------------------------------------------------

static void
run(std::vector<Magick::Image> &images, const Config &config,
	Ort::Session &session, std::vector<int64_t> shape)
{
	auto batch = shape[0] = images.size();

	Ort::AllocatorWithDefaultOptions allocator;
	auto tensor = Ort::Value::CreateTensor<float>(
		allocator, shape.data(), shape.size());

	auto input_len = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
	auto input_data = tensor.GetTensorMutableData<float>(), pi = input_data;
	for (int64_t i = 0; i < batch; i++) {
		switch (config.shape) {
		case Config::Shape::NCHW:
			pi = image_to_nchw(pi, images.at(i), config.channels);
			break;
		case Config::Shape::NHWC:
			pi = image_to_nhwc(pi, images.at(i), config.channels);
		}
	}
	if (config.normalize) {
		pi = input_data;
		for (size_t i = 0; i < input_len; i++)
			*pi++ /= 255.0;
	}

	std::string input_name =
		session.GetInputNameAllocated(0, allocator).get();
	std::string output_name =
		session.GetOutputNameAllocated(0, allocator).get();

	std::vector<const char *> input_names = {input_name.c_str()};
	std::vector<const char *> output_names = {output_name.c_str()};

	auto outputs = session.Run(Ort::RunOptions{},
		input_names.data(), &tensor, input_names.size(),
		output_names.data(), output_names.size());
	if (outputs.size() != 1 || !outputs[0].IsTensor()) {
		fprintf(stderr, "Wrong output\n");
		return;
	}

	auto output_len = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	auto output_data = outputs.front().GetTensorData<float>(), po = output_data;
	if (output_len != batch * config.tags.size()) {
		fprintf(stderr, "Tags don't match the output\n");
		return;
	}

	for (size_t i = 0; i < batch; i++) {
		for (size_t t = 0; t < config.tags.size(); t++) {
			float value = *po++;
			if (config.sigmoid)
				value = 1 / (1 + std::exp(-value));
			if (value > g.threshold) {
				printf("%s\t%.2f\t%s\n", images.at(i).fileName().c_str(),
					value, config.tags.at(t).c_str());
			}
		}
	}
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static void
parse_options(const std::string &options)
{
	auto semicolon = options.find(";");
	auto name = options.substr(0, semicolon);
	auto sequence = options.substr(semicolon);

	std::map<std::string, std::string> kv;
	std::regex re(R"(;*([^;=]+)=([^;=]+))", std::regex::optimize);
	std::sregex_iterator it(sequence.begin(), sequence.end(), re), end;
	for (; it != end; ++it)
		kv[it->str(1)] = it->str(2);
	g.options.insert_or_assign(name, std::move(kv));
}

static std::tuple<std::vector<const char *>, std::vector<const char *>>
unpack_options(const std::string &provider)
{
	std::vector<const char *> keys, values;
	if (g.options.count(provider)) {
		for (const auto &kv : g.options.at(provider)) {
			keys.push_back(kv.first.c_str());
			values.push_back(kv.second.c_str());
		}
	}
	return {keys, values};
}

static void
add_providers(Ort::SessionOptions &options)
{
	auto api = Ort::GetApi();
	auto v_providers = Ort::GetAvailableProviders();
	std::set<std::string> providers(v_providers.begin(), v_providers.end());

	if (g.debug) {
		printf("Providers:");
		for (const auto &it : providers)
			printf(" %s", it.c_str());
		printf("\n");
	}

	// There is a string-based AppendExecutionProvider() method,
	// but it cannot be used with all providers.
	// TODO: Make it possible to disable providers.
	// TODO: Providers will deserve some performance tuning.

	if (g.cpu)
		return;

#ifdef __APPLE__
	if (providers.count("CoreMLExecutionProvider")) {
		try {
			Ort::ThrowOnError(
				OrtSessionOptionsAppendExecutionProvider_CoreML(options, 0));
		} catch (const std::exception &e) {
			fprintf(stderr, "CoreML unavailable: %s\n", e.what());
		}
	}
#endif

#if TENSORRT
	// TensorRT should be the more performant execution provider, however:
	//  - it is difficult to set up (needs logging in to download),
	//  - with WD v1.4 ONNX models, one gets "Your ONNX model has been generated
	//    with INT64 weights, while TensorRT does not natively support INT64.
	//    Attempting to cast down to INT32." and that's not nice.
	if (providers.count("TensorrtExecutionProvider")) {
		OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;
		Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
		auto [keys, values] = unpack_options("TensorrtExecutionProvider");
		if (!keys.empty()) {
			Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
				tensorrt_options, keys.data(), values.data(), keys.size()));
		}

		try {
			options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
		} catch (const std::exception &e) {
			fprintf(stderr, "TensorRT unavailable: %s\n", e.what());
		}
		api.ReleaseTensorRTProviderOptions(tensorrt_options);
	}
#endif

	// See CUDA-ExecutionProvider.html for documentation.
	if (providers.count("CUDAExecutionProvider")) {
		OrtCUDAProviderOptionsV2* cuda_options = nullptr;
		Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
		auto [keys, values] = unpack_options("CUDAExecutionProvider");
		if (!keys.empty()) {
			Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
				cuda_options, keys.data(), values.data(), keys.size()));
		}

		try {
			options.AppendExecutionProvider_CUDA_V2(*cuda_options);
		} catch (const std::exception &e) {
			fprintf(stderr, "CUDA unavailable: %s\n", e.what());
		}
		api.ReleaseCUDAProviderOptions(cuda_options);
	}

	if (providers.count("ROCMExecutionProvider")) {
		OrtROCMProviderOptions rocm_options = {};
		auto [keys, values] = unpack_options("ROCMExecutionProvider");
		if (!keys.empty()) {
			Ort::ThrowOnError(api.UpdateROCMProviderOptions(
				&rocm_options, keys.data(), values.data(), keys.size()));
		}

		try {
			options.AppendExecutionProvider_ROCM(rocm_options);
		} catch (const std::exception &e) {
			fprintf(stderr, "ROCM unavailable: %s\n", e.what());
		}
	}

	// The CPU provider is the default fallback, if everything else fails.
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static std::string
print_shape(const Ort::ConstTensorTypeAndShapeInfo &info)
{
	std::vector<const char *> names(info.GetDimensionsCount());
	info.GetSymbolicDimensions(names.data(), names.size());

	auto shape = info.GetShape();
	std::string result;
	for (size_t i = 0; i < shape.size(); i++) {
		if (shape[i] < 0)
			result.append(names.at(i));
		else
			result.append(std::to_string(shape[i]));
		result.append(" x ");
	}
	if (!result.empty())
		result.erase(result.size() - 3);
	return result;
}

static void
print_shapes(const Ort::Session &session)
{
	Ort::AllocatorWithDefaultOptions allocator;
	for (size_t i = 0; i < session.GetInputCount(); i++) {
		std::string name = session.GetInputNameAllocated(i, allocator).get();
		auto info = session.GetInputTypeInfo(i);
		auto shape = print_shape(info.GetTensorTypeAndShapeInfo());
		printf("Input: %s: %s\n", name.c_str(), shape.c_str());
	}
	for (size_t i = 0; i < session.GetOutputCount(); i++) {
		std::string name = session.GetOutputNameAllocated(i, allocator).get();
		auto info = session.GetOutputTypeInfo(i);
		auto shape = print_shape(info.GetTensorTypeAndShapeInfo());
		printf("Output: %s: %s\n", name.c_str(), shape.c_str());
	}
}

static void
infer(Ort::Env &env, const char *path, const std::vector<std::string> &images)
{
	Config config;
	read_config(config, path);

	Ort::SessionOptions session_options;
	add_providers(session_options);

	Ort::Session session = Ort::Session(env,
		std::filesystem::path(path).replace_extension("onnx").c_str(),
		session_options);

	if (g.debug)
		print_shapes(session);

	if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
		fprintf(stderr, "Invalid input or output shape\n");
		exit(EXIT_FAILURE);
	}

	auto input_info = session.GetInputTypeInfo(0);
	auto shape = input_info.GetTensorTypeAndShapeInfo().GetShape();
	if (shape.size() != 4) {
		fprintf(stderr, "Incompatible input tensor format\n");
		exit(EXIT_FAILURE);
	}
	if (shape.at(0) > 1) {
		fprintf(stderr, "Fixed batching not supported\n");
		exit(EXIT_FAILURE);
	}
	if (shape.at(0) >= 0 && g.batch > 1) {
		fprintf(stderr, "Requested batching for a non-batching model\n");
		exit(EXIT_FAILURE);
	}

	int64_t *height = {}, *width = {}, *channels = {};
	switch (config.shape) {
	case Config::Shape::NCHW:
		channels = &shape[1];
		height = &shape[2];
		width = &shape[3];
		break;
	case Config::Shape::NHWC:
		height = &shape[1];
		width = &shape[2];
		channels = &shape[3];
		break;
	}

	// Variable dimensions don't combine well with batches.
	if (*height < 0)
		*height = config.size;
	if (*width < 0)
		*width = config.size;
	if (*channels != 3 || *height < 1 || *width < 1) {
		fprintf(stderr, "Incompatible input tensor format\n");
		return;
	}

	// TODO: Image loading is heavily parallelizable. In theory.
	std::vector<Magick::Image> batch;
	for (const auto &filename : images) {
		Magick::Image image;
		try {
			image = load(filename, config, *width, *height);
		} catch (const std::exception &e) {
			fprintf(stderr, "%s: %s\n", filename.c_str(), e.what());
			continue;
		}

		if (*height != image.rows() || *width != image.columns()) {
			fprintf(stderr, "%s: %s\n", filename.c_str(), "tensor mismatch");
			continue;
		}

		batch.push_back(image);
		if (batch.size() == g.batch) {
			run(batch, config, session, shape);
			batch.clear();
		}
	}
	if (!batch.empty())
		run(batch, config, session, shape);
}

int
main(int argc, char *argv[])
{
	auto invocation_name = argv[0];
	auto print_usage = [=] {
		fprintf(stderr,
			"Usage: %s [-b BATCH] [--cpu] [-d] [-o EP;KEY=VALUE...] "
			"[-t THRESHOLD] MODEL { --pipe | [IMAGE...] }\n", invocation_name);
	};

	static option opts[] = {
		{"batch", required_argument, 0, 'b'},
		{"cpu", no_argument, 0, 'c'},
		{"debug", no_argument, 0, 'd'},
		{"help", no_argument, 0, 'h'},
		{"options", required_argument, 0, 'o'},
		{"pipe", no_argument, 0, 'p'},
		{"threshold", required_argument, 0, 't'},
		{nullptr, 0, 0, 0},
	};

	bool pipe = false;
	while (1) {
		int option_index = 0;
		auto c = getopt_long(argc, const_cast<char *const *>(argv),
			"b:cdho:pt:", opts, &option_index);
		if (c == -1)
			break;

		char *end = nullptr;
		switch (c) {
		case 'b':
			errno = 0, g.batch = strtol(optarg, &end, 10);
			if (errno || *end || g.batch < 1 || g.batch > SHRT_MAX) {
				fprintf(stderr, "Batch size must be a positive number\n");
				exit(EXIT_FAILURE);
			}
			break;
		case 'c':
			g.cpu = true;
			break;
		case 'd':
			g.debug++;
			break;
		case 'h':
			print_usage();
			return 0;
		case 'o':
			parse_options(optarg);
			break;
		case 'p':
			pipe = true;
			break;
		case 't':
			errno = 0, g.threshold = strtod(optarg, &end);
			if (errno || *end || !std::isfinite(g.threshold) ||
				g.threshold < 0 || g.threshold > 1) {
				fprintf(stderr, "Threshold must be a number within 0..1\n");
				exit(EXIT_FAILURE);
			}
			break;
		default:
			print_usage();
			return 1;
		}
	}

	argv += optind;
	argc -= optind;

	// TODO: There's actually no need to slurp all the lines up front.
	std::vector<std::string> paths;
	if (pipe) {
		if (argc != 1) {
			print_usage();
			return 1;
		}

		std::string line;
		while (std::getline(std::cin, line))
			paths.push_back(line);
	} else {
		if (argc < 1) {
			print_usage();
			return 1;
		}

		paths.assign(argv + 1, argv + argc);
	}

	// XXX: GraphicsMagick initializes signal handlers here,
	// one needs to use MagickLib::InitializeMagickEx()
	// with MAGICK_OPT_NO_SIGNAL_HANDER to prevent that.
	//
	// ImageMagick conveniently has the opposite default.
	//
	// Once processing images in parallel, consider presetting
	// OMP_NUM_THREADS=1 (GM) and/or MAGICK_THREAD_LIMIT=1 (IM).
	Magick::InitializeMagick(nullptr);

	OrtLoggingLevel logging = g.debug > 1
		? ORT_LOGGING_LEVEL_VERBOSE
		: ORT_LOGGING_LEVEL_WARNING;

	// Creating an environment before initializing providers in order to avoid:
	// "Attempt to use DefaultLogger but none has been registered."
	Ort::Env env(logging, invocation_name);
	infer(env, argv[0], paths);
	return 0;
}
