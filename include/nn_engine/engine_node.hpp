#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nn_engine/engine_wrapper.hpp"

// Include custom message and service types.
#include "nn_engine/msg/object_detection_output.hpp"
#include "nn_engine/srv/change_model_rate.hpp"
#include "nn_engine/srv/get_model_names.hpp"

// For keeping bounding boxes as a matrix.
#include "std_msgs/msg/float32_multi_array.hpp"

class EngineNode : public rclcpp::Node
{
public:
  EngineNode();

private:
  // Callback for incoming image messages.
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

  // Helper conversion functions.
  std::vector<int32_t> convertCvMatToIntVector(const cv::Mat& mat);
  std::vector<float> convertCvMatToFloatVector(const cv::Mat& mat);
  std_msgs::msg::Float32MultiArray convertCvMatToFloatMultiArray(const cv::Mat& mat);

  // Engine instance.
  std::unique_ptr<EngineWrapper> engine_;

  // Subscriber for input images.
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  // Timers for periodically publishing each model head's output.
  std::vector<rclcpp::TimerBase::SharedPtr> timers_;

  // Service servers.
  rclcpp::Service<nn_engine::srv::ChangeModelRate>::SharedPtr change_rate_srv_;
  rclcpp::Service<nn_engine::srv::GetModelNames>::SharedPtr get_model_names_srv_;
};