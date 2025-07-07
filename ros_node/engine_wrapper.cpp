#include "vp_engine/engine_wrapper.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>

// ROS2 includes for image messages and conversion.
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;

// Conversion functions taken from: https://gist.github.com/aFewThings/c79e124f649ea9928bfc7bb8827f1a1c

// Helper function to convert a cv::Mat to a NumPy array.
np::ndarray ConvertMatToNDArray(const cv::Mat& mat) {
	bp::tuple shape = bp::make_tuple(mat.rows, mat.cols, mat.channels());
	bp::tuple stride = bp::make_tuple(mat.channels() * mat.cols * sizeof(uchar), mat.channels() * sizeof(uchar), sizeof(uchar));
	np::dtype dt = np::dtype::get_builtin<uchar>();
	np::ndarray ndImg = np::from_data(mat.data, dt, shape, stride, bp::object());

	return ndImg;
}

// Helper function to convert a NumPy array to a cv::Mat.
cv::Mat ConvertNDArrayToMat(const np::ndarray& ndarr) {
	//int length = ndarr.get_nd(); // get_nd() returns num of dimensions. this is used as a length, but we don't need to use in this case. because we know that image has 3 dimensions.
	const Py_intptr_t* shape = ndarr.get_shape(); // get_shape() returns Py_intptr_t* which we can get the size of n-th dimension of the ndarray.
	char* dtype_str = bp::extract<char *>(bp::str(ndarr.get_dtype()));

	// variables for creating Mat object
	int rows = shape[0];
	int cols = shape[1];
	int channel = shape[2];
	int depth;

	// you should find proper type for c++. in this case we use 'CV_8UC3' image, so we need to create 'uchar' type Mat.
	if (!strcmp(dtype_str, "uint8")) {
		depth = CV_8U;
	}
	else {
		std::cout << "wrong dtype error" << std::endl;
		return cv::Mat();
	}

	int type = CV_MAKETYPE(depth, channel); // CV_8UC3

	cv::Mat mat = cv::Mat(rows, cols, type);
	memcpy(mat.data, ndarr.get_data(), sizeof(uchar) * rows * cols * channel);

	return mat;
}

// Implementation of PyGILGuard
PyGILGuard::PyGILGuard() {
  gstate_ = PyGILState_Ensure();
}

PyGILGuard::~PyGILGuard() {
  PyGILState_Release(gstate_);
}

// Implementation of EngineWrapper constructor and destructor

EngineWrapper::EngineWrapper(const std::string& package_dir,
                             const std::string& config_file,
                             const std::string& registry_file)
{
  PyGILGuard guard;
  try {
    // Initialize Python and NumPy if needed.
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }
    np::initialize();

    // Insert package_dir into sys.path.
    bp::object sys_module = bp::import("sys");
    bp::object sys_path = sys_module.attr("path");
    sys_path.attr("insert")(0, package_dir);

    // Import the 'src' module and get the Engine class.
    bp::object src_module = bp::import("src");
    bp::object engine_cls = src_module.attr("Engine");

    // Create an engine instance (with embedded_in_ros2 set to true).
    engine_ = engine_cls(config_file, registry_file, true);

    // Call build and start_inference on the engine instance.
    engine_.attr("build")();
    engine_.attr("start_inference")();
    engine_.attr("test")(10.0); 
  }
  catch (bp::error_already_set&) {
    PyErr_Print();
  }
}

EngineWrapper::~EngineWrapper() {
  PyGILGuard guard;
  try {
    engine_.attr("stop")();
  }
  catch (bp::error_already_set&) {
    PyErr_Print();
  }
}

bool EngineWrapper::change_model_rate(const std::string& model_name, double new_rate) {
  return executePythonCall([&]() -> bool {
    bp::object result = engine_.attr("change_model_rate")(model_name, new_rate);
    return bp::extract<bool>(result);
  });
}

FoundationModelParams EngineWrapper::getFoundationModelParams() {
  return executePythonCall([&]() -> FoundationModelParams {
    bp::dict result = bp::extract<bp::dict>(engine_.attr("get_foundation_model_params")());
    FoundationModelParams params;
    params.name = bp::extract<std::string>(result["name"]);
    params.rate = bp::extract<double>(result["rate"]);
    return params;
  });
}

std::vector<ModelHeadParams> EngineWrapper::getModelHeadsParams() {
  return executePythonCall([&]() -> std::vector<ModelHeadParams> {
    bp::list py_list = bp::extract<bp::list>(engine_.attr("get_model_heads_params")());
    std::vector<ModelHeadParams> heads;
    for (int i = 0; i < len(py_list); i++) {
      bp::dict dict = bp::extract<bp::dict>(py_list[i]);
      ModelHeadParams head_params;
      head_params.name = bp::extract<std::string>(dict["name"]);
      head_params.rate = bp::extract<double>(dict["rate"]);
      head_params.output_type = bp::extract<std::string>(dict["output_type"]);
      heads.push_back(head_params);
    }
    return heads;
  });
}

bool EngineWrapper::inputImage(const sensor_msgs::msg::Image::SharedPtr& image_msg) {
  return executePythonCall([&]() -> bool {
    try {
      // Convert the ROS2 Image message to a cv::Mat (assuming "bgr8" encoding).
      cv::Mat cv_img = cv_bridge::toCvShare(image_msg, "bgr8")->image;
      // Convert cv::Mat to a NumPy array using the helper function.
      np::ndarray np_img = ConvertMatToNDArray(cv_img);
      
      // Create a timestamp in milliseconds.
      double timestamp_ms = image_msg->header.stamp.sec * 1000.0 +
                            image_msg->header.stamp.nanosec / 1e6;
      
      // Call the Python engine's input_image method directly, passing the millisecond timestamp.
      bp::object result = engine_.attr("input_image")(np_img, timestamp_ms);
      return bp::extract<bool>(result);
    } catch (cv_bridge::Exception &e) {
      return false;
    }
  });
}

cv::Mat EngineWrapper::getHeadImageOutput(int head_id) {
  return executePythonCall([&]() -> cv::Mat {
    bp::object result = engine_.attr("get_head_output")(head_id);
    if (result.ptr() == Py_None) {
      return cv::Mat(); // Use .empty() to check for an empty Mat.
    }
    np::ndarray np_img = bp::extract<np::ndarray>(result);
    return ConvertNDArrayToMat(np_img);
  });
}

ObjectDetectionOutput EngineWrapper::getHeadObjectDetectionOutput(int head_id) {
  return executePythonCall([&]() -> ObjectDetectionOutput {
    ObjectDetectionOutput output;  // All cv::Mat fields are initially empty.
    
    bp::object result = engine_.attr("get_head_output")(head_id);
    if (result.ptr() == Py_None) {
      return output;  // Return empty output if Python returns None.
    }
    
    bp::list result_list = bp::extract<bp::list>(result);
    if (len(result_list) != 3) {
      std::runtime_error("Expected a list of 3 NumPy arrays for object detection output.");
    }
    
    // Process element 0: Labels (float16 -> int)
    {
      np::ndarray labels_array = bp::extract<np::ndarray>(result_list[0]);
      const Py_intptr_t* shape = labels_array.get_shape();
      int N = shape[0];
      // Create a temporary cv::Mat view of the labels assuming CV_16F.
      cv::Mat temp_labels(N, 1, CV_16F, labels_array.get_data());
      temp_labels.convertTo(output.labels, CV_32S);  // Convert to 32-bit integer.
    }
    
    // Process element 1: Scores (float16 -> float)
    {
      np::ndarray scores_array = bp::extract<np::ndarray>(result_list[1]);
      const Py_intptr_t* shape = scores_array.get_shape();
      int N = shape[0];
      cv::Mat temp_scores(N, 1, CV_16F, scores_array.get_data());
      temp_scores.convertTo(output.scores, CV_32F);  // Convert to 32-bit float.
    }
    
    // Process element 2: Normalized Bounding Boxes (float16 -> float)
    {
      np::ndarray boxes_array = bp::extract<np::ndarray>(result_list[2]);
      const Py_intptr_t* shape = boxes_array.get_shape();
      int N = shape[0];
      int cols = (boxes_array.get_nd() > 1) ? shape[1] : 1;  // Expecting 4 columns, for example.
      cv::Mat temp_boxes(N, cols, CV_16F, boxes_array.get_data());
      temp_boxes.convertTo(output.boxes, CV_32F);  // Convert to 32-bit float.
    }
    
    return output;
  });
}