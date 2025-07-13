#include <Python.h>
#include <boost/python.hpp>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// ROS2 includes for image messages and conversion.
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

namespace bp = boost::python;

// RAII guard for Python GIL management.
class PyGILGuard {
public:
  PyGILGuard();
  ~PyGILGuard();
private:
  PyGILState_STATE gstate_;
};

// Helper function template that wraps a callable with GIL management and error handling.
template<typename Func>
auto executePythonCall(Func&& func) -> decltype(func()) {
  PyGILGuard guard;
  try {
    return func();
  }
  catch (bp::error_already_set&) {
    PyErr_Print();
    throw;
  }
}

/**
 * @brief Structure representing the parameters of the foundation model.
 */
struct FoundationModelParams {
    std::string name;
    double rate;
};

/**
 * @brief Structure representing the parameters of a model head.
 */
struct ModelHeadParams {
    std::string name;
    double rate;
    std::string output_type;
};

/**
 * @brief Structure representing the output of object detection
 */
struct ObjectDetectionOutput {
  cv::Mat labels;  // Expected to be a single-column matrix of integers.
  cv::Mat scores;  // Single-column matrix of floats.
  cv::Mat boxes;   // Matrix of floats (e.g., N×4 for bounding boxes).
};

/**
 * @brief A C++ wrapper for the Python Engine.
 *
 * This class sets up the Python environment and wraps the Engine class.
 */
class EngineWrapper {
public:
  /**
   * @brief Constructor that sets up the Python environment, imports the engine module,
   * and starts the engine by calling build() and start_inference().
   * 
   * @param package_dir The directory to insert into sys.path so Python can find your modules.
   * @param config_file Path to the engine configuration file.
   * @param registry_file Path to the engine registry file.
   */
  EngineWrapper(const std::string& package_dir,
                const std::string& config_file,
                const std::string& registry_file);
  
  ~EngineWrapper();

  /// Changes the inference rate of the given model. Returns true on success.
  bool change_model_rate(const std::string& model_name, double new_rate);

  /// Gets the parameters of the foundation model as a C++ object.
  FoundationModelParams getFoundationModelParams();

  /// Gets the parameters of the model heads as a vector of C++ objects.
  std::vector<ModelHeadParams> getModelHeadsParams();

  /**
   * @brief Inputs a ROS2 Image message into the engine.
   *
   * This method converts the sensor_msgs::msg::Image (using cv_bridge)
   * into an OpenCV image and then calls inputImage().
   *
   * @param image_msg Shared pointer to the ROS2 Image message.
   * @param image_id Optional image identifier (default: -1).
   * @return True if the image was successfully input, false otherwise.
   */
  bool inputImage(const sensor_msgs::msg::Image::SharedPtr& image_msg);

  /**
   * @brief Get the image output from the engine for a specific head.
   *
   * This method calls the Python engine's get_head_output and expects a NumPy array,
   * which it converts to a cv::Mat.
   *
   * @param head_id The index of the head.
   * @return cv::Mat containing the output image. Returns an empty cv::Mat on error.
   */
  cv::Mat getHeadImageOutput(int head_id);

  /**
   * @brief Get the object detection output from the engine for a specific head.
   *
   * This method calls the Python engine's get_head_output and expects a list of NumPy arrays,
   * which it converts into a vector of cv::Mat (e.g. labels, scores, boxes).
   *
   * @param head_id The index of the head.
   * @return ObjectDetectionOutput containing the detection outputs in order. If an error occurs,
   *         an empty vector is returned.
   */
  ObjectDetectionOutput getHeadObjectDetectionOutput(int head_id);

private:
  bp::object engine_; // The wrapped Python Engine instance.
};