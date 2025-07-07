#include "vp_engine/engine_node.hpp"
#include <filesystem>
#include <chrono>
#include <functional>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/msg/image.hpp"

using namespace std::chrono_literals;

EngineNode::EngineNode()
: Node("engine_node")
{
  // Declare read-only parameters.
  auto config_desc = rcl_interfaces::msg::ParameterDescriptor{};
  config_desc.read_only = true;
  this->declare_parameter<std::string>("engine_configuration_filepath", config_desc);

  auto registry_desc = rcl_interfaces::msg::ParameterDescriptor{};
  registry_desc.read_only = true;
  this->declare_parameter<std::string>("model_registry_filepath", registry_desc);

  auto topic_name = rcl_interfaces::msg::ParameterDescriptor{};
  topic_name.read_only = true;
  this->declare_parameter<std::string>("input_topic_name", topic_name);

  auto publisher_refresh_rate = rcl_interfaces::msg::ParameterDescriptor{};
  publisher_refresh_rate.read_only = true;
  this->declare_parameter<double>("publishers_refresh_rate", 200.0, publisher_refresh_rate);

  // Retrieve parameter values.
  std::string config_file = this->get_parameter("engine_configuration_filepath").as_string();
  std::string registry_file = this->get_parameter("model_registry_filepath").as_string();
  std::string input_topic_name = this->get_parameter("input_topic_name").as_string();
  double refresh_rate = this->get_parameter("publishers_refresh_rate").as_double();

  // Get the package directory (assumes two parent directories above this file).
  std::filesystem::path filePath = __FILE__;
  std::string package_dir = filePath.parent_path().parent_path().string();

  config_file = package_dir + "/" + config_file;
  registry_file = package_dir + "/" + registry_file;

  // Create the engine instance using the provided parameters.
  engine_ = std::make_unique<EngineWrapper>(package_dir, config_file, registry_file);

  // Initialize subscriber for input images.
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      input_topic_name, 10,
      std::bind(&EngineNode::imageCallback, this, std::placeholders::_1));

  // Get model head parameters from the engine.
  auto heads = engine_->getModelHeadsParams();

  // For each model head, create a publisher and a timer based on the head’s output type.
  for (size_t i = 0; i < heads.size(); ++i) {
    const auto & head = heads[i];
    double period_sec = refresh_rate > 0 ? 1.0 / refresh_rate : 0.005;

    if (head.output_type == "image") {
      // Publisher for image output.
      auto pub = this->create_publisher<sensor_msgs::msg::Image>(head.name, 10);
      auto timer = this->create_wall_timer(
          std::chrono::duration<double>(period_sec),
          [this, i, pub]() {
            cv::Mat output_img = engine_->getHeadImageOutput(i);
            if (output_img.empty()) {
              return;
            }
            // Convert cv::Mat to ROS image message.
            std_msgs::msg::Header header;
            header.stamp = this->get_clock()->now();
            auto msg = cv_bridge::CvImage(header, "bgr8", output_img).toImageMsg();
            pub->publish(*msg);
          });
      timers_.push_back(timer);
    } else if (head.output_type == "object_detection") {
      // Publisher for object detection output.
      auto pub = this->create_publisher<vp_engine::msg::ObjectDetectionOutput>(head.name, 10);
      auto timer = this->create_wall_timer(
          std::chrono::duration<double>(period_sec),
          [this, i, pub]() {
            // Retrieve the detection output from the engine.
            ObjectDetectionOutput od_output = engine_->getHeadObjectDetectionOutput(i);
            if (od_output.boxes.empty()) {
              return;
            }
            vp_engine::msg::ObjectDetectionOutput msg;
            msg.header.stamp = this->get_clock()->now();
            // Convert cv::Mat outputs to STL vectors.
            msg.labels = convertCvMatToIntVector(od_output.labels);
            msg.scores = convertCvMatToFloatVector(od_output.scores);
            // Convert the bounding boxes matrix to a Float32MultiArray.
            msg.boxes  = convertCvMatToFloatMultiArray(od_output.boxes);
            pub->publish(msg);
          }); 
      timers_.push_back(timer);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Unknown output type '%s' for head %zu", head.output_type.c_str(), i);
      std::runtime_error("Unknown output type for head " + std::to_string(i));
    }
  }

  // Service to change a model's firing rate.
  change_rate_srv_ = this->create_service<vp_engine::srv::ChangeModelRate>(
      "change_model_rate",
      [this](const std::shared_ptr<vp_engine::srv::ChangeModelRate::Request> req,
             std::shared_ptr<vp_engine::srv::ChangeModelRate::Response> res) {
        bool success = engine_->change_model_rate(req->model_name, req->new_rate);
        res->success = success;
        if (success) {
          RCLCPP_INFO(this->get_logger(), "Changed rate for model %s to %f",
                      req->model_name.c_str(), req->new_rate);
        } else {
          RCLCPP_ERROR(this->get_logger(), "Failed to change rate for model %s",
                       req->model_name.c_str());
        }
      });

  // Service to retrieve all model names.
  get_model_names_srv_ = this->create_service<vp_engine::srv::GetModelNames>(
      "get_model_names",
      [this](const std::shared_ptr<vp_engine::srv::GetModelNames::Request> /*req*/,
             std::shared_ptr<vp_engine::srv::GetModelNames::Response> res) {
        // Include the foundation model name as well as each model head name.
        auto foundation = engine_->getFoundationModelParams();
        auto heads = engine_->getModelHeadsParams();
        res->model_names.push_back(foundation.name);
        for (const auto & head : heads) {
          res->model_names.push_back(head.name);
        }
      });
}

void EngineNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // Input the image into the engine.
  if (!engine_->inputImage(msg)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to input image to the engine.");
  }
}

std::vector<int32_t> EngineNode::convertCvMatToIntVector(const cv::Mat& mat)
{
  std::vector<int32_t> vec;
  if (!mat.empty() && mat.isContinuous()) {
    vec.assign(reinterpret_cast<const int32_t*>(mat.datastart),
               reinterpret_cast<const int32_t*>(mat.dataend));
  }
  return vec;
}

std::vector<float> EngineNode::convertCvMatToFloatVector(const cv::Mat& mat)
{
  std::vector<float> vec;
  if (!mat.empty() && mat.isContinuous()) {
    vec.assign(reinterpret_cast<const float*>(mat.datastart),
               reinterpret_cast<const float*>(mat.dataend));
  }
  return vec;
}

std_msgs::msg::Float32MultiArray EngineNode::convertCvMatToFloatMultiArray(const cv::Mat& mat)
{
  std_msgs::msg::Float32MultiArray multi_array;
  // Ensure the matrix is not empty, has 4 columns, and is of type float.
  if (mat.empty() || mat.cols != 4 || mat.type() != CV_32F) {
    RCLCPP_ERROR(this->get_logger(), "Invalid bounding boxes matrix dimensions or type.");
    return multi_array;
  }
  
  // Set up layout for a 2D matrix with dimensions [rows, 4].
  multi_array.layout.dim.resize(2);
  multi_array.layout.dim[0].label = "rows";
  multi_array.layout.dim[0].size = mat.rows;
  multi_array.layout.dim[0].stride = mat.rows * mat.cols;
  multi_array.layout.dim[1].label = "cols";
  multi_array.layout.dim[1].size = mat.cols;
  multi_array.layout.dim[1].stride = mat.cols;
  multi_array.layout.data_offset = 0;
  
  // Fill data from the cv::Mat.
  if (mat.isContinuous()) {
    multi_array.data.assign(reinterpret_cast<const float*>(mat.datastart),
                              reinterpret_cast<const float*>(mat.dataend));
  } else {
    for (int i = 0; i < mat.rows; ++i) {
      const float* row_ptr = mat.ptr<float>(i);
      multi_array.data.insert(multi_array.data.end(), row_ptr, row_ptr + mat.cols);
    }
  }
  return multi_array;
}