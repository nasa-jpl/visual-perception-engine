#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <cctype>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;

class ImagePublisher : public rclcpp::Node
{
public:
  ImagePublisher()
  : Node("image_publisher"), current_index_(0)
  {
    // Declare parameters with default values
    this->declare_parameter("image_directory", "tests/resources/inputs,tests/resources/object_detection/inputs");
    this->declare_parameter("image_topic", "images");
    this->declare_parameter("publish_frequency", 30.0);

    // Get parameter values
    std::string image_directory;
    std::string image_topic;
    double publish_frequency;
    this->get_parameter("image_directory", image_directory);
    this->get_parameter("image_topic", image_topic);
    this->get_parameter("publish_frequency", publish_frequency);

    // Compute package directory based on __FILE__
    std::filesystem::path filePath = __FILE__;
    std::string package_dir = filePath.parent_path().parent_path().string();

    // Split the image_directory parameter by comma to support multiple directories.
    std::vector<std::string> directories;
    std::stringstream ss(image_directory);
    std::string token;
    while (std::getline(ss, token, ',')) {
      // Trim leading and trailing whitespace
      token.erase(token.begin(), std::find_if(token.begin(), token.end(),
          [](unsigned char ch) { return !std::isspace(ch); }));
      token.erase(std::find_if(token.rbegin(), token.rend(),
          [](unsigned char ch) { return !std::isspace(ch); }).base(), token.end());
      // If the token is a relative path, prepend the package directory
      if (!std::filesystem::path(token).is_absolute()) {
        token = package_dir + "/" + token;
      }
      directories.push_back(token);
    }

    // Create a publisher with the given topic name
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>(image_topic, 10);

    // Loop over each provided directory and load images
    for (const auto & dir : directories) {
      if (!std::filesystem::exists(dir)) {
        RCLCPP_WARN(this->get_logger(), "Directory does not exist: %s", dir.c_str());
        continue;
      }
      for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
          auto ext = entry.path().extension().string();
          // Convert extension to lower-case for consistency
          std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
          if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp") {
            cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (!image.empty()) {
              // Resize the image to 1920x1080 (width x height)
              cv::Mat resized;
              cv::resize(image, resized, cv::Size(1920, 1080));
              images_.push_back(resized);
              RCLCPP_INFO(this->get_logger(), "Loaded and resized image: %s", entry.path().string().c_str());
            } else {
              RCLCPP_WARN(this->get_logger(), "Failed to load image: %s", entry.path().string().c_str());
            }
          }
        }
      }
    }

    if (images_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "No images loaded from the provided directories.");
      rclcpp::shutdown();
      return;
    }

    // Calculate timer period based on the publish frequency (Hz)
    int period_ms = static_cast<int>(1000.0 / publish_frequency);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(period_ms),
      std::bind(&ImagePublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    // Get the current image from the vector.
    cv::Mat cv_image = images_[current_index_];

    // Convert the OpenCV image to a ROS2 Image message using cv_bridge.
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_image).toImageMsg();
    msg->header.stamp = this->now();
    publisher_->publish(*msg);

    RCLCPP_INFO(this->get_logger(), "Publishing image %zu", current_index_);

    // Cycle to the next image in a loop.
    current_index_ = (current_index_ + 1) % images_.size();
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::vector<cv::Mat> images_;
  size_t current_index_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImagePublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}