#include "rclcpp/rclcpp.hpp"
#include <Python.h>
#include <memory>
#include <boost/python.hpp>

#include "vp_engine/engine_node.hpp"

namespace bp = boost::python;

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Initialize the Python interpreter manually
  Py_Initialize();

  {
    auto node = std::make_shared<EngineNode>();
    rclcpp::spin(node);
  } // Python-dependent objects go out of scope here and thus are destroyed.

  // Finalize the Python interpreter.
  Py_FinalizeEx();

  rclcpp::shutdown();
  return 0;
}