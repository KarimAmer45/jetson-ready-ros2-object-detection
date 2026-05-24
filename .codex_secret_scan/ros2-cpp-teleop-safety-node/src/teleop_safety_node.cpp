#include "ros2_cpp_teleop_safety_node/teleop_safety_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace ros2_cpp_teleop_safety_node
{

TeleopSafetyNode::TeleopSafetyNode(const rclcpp::NodeOptions & options)
: Node("teleop_safety_node", options)
{
  loadParameters();

  command_pub_ = create_publisher<geometry_msgs::msg::Twist>(params_.output_topic, 10);

  command_sub_ = create_subscription<geometry_msgs::msg::Twist>(
    params_.input_topic,
    10,
    [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
      handleCommand(msg);
    });

  scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
    params_.scan_topic,
    rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
      handleScan(msg);
    });

  estop_sub_ = create_subscription<std_msgs::msg::Bool>(
    params_.estop_topic,
    10,
    [this](const std_msgs::msg::Bool::SharedPtr msg) {
      handleEstop(msg);
    });

  speed_limit_sub_ = create_subscription<std_msgs::msg::Float64>(
    params_.speed_limit_topic,
    10,
    [this](const std_msgs::msg::Float64::SharedPtr msg) {
      handleSpeedLimit(msg);
    });

  watchdog_timer_ = create_wall_timer(
    std::chrono::duration<double>(params_.timer_period),
    [this]() {
      handleWatchdog();
    });

  RCLCPP_INFO(
    get_logger(),
    "Teleop safety node guarding '%s' -> '%s'",
    params_.input_topic.c_str(),
    params_.output_topic.c_str());
}

void TeleopSafetyNode::loadParameters()
{
  params_.input_topic = declare_parameter<std::string>("input_topic", params_.input_topic);
  params_.output_topic = declare_parameter<std::string>("output_topic", params_.output_topic);
  params_.scan_topic = declare_parameter<std::string>("scan_topic", params_.scan_topic);
  params_.estop_topic = declare_parameter<std::string>("estop_topic", params_.estop_topic);
  params_.speed_limit_topic =
    declare_parameter<std::string>("speed_limit_topic", params_.speed_limit_topic);
  params_.command_timeout = declare_parameter<double>("command_timeout", params_.command_timeout);
  params_.timer_period = declare_parameter<double>("timer_period", params_.timer_period);
  params_.max_linear_speed = declare_parameter<double>("max_linear_speed", params_.max_linear_speed);
  params_.max_lateral_speed =
    declare_parameter<double>("max_lateral_speed", params_.max_lateral_speed);
  params_.max_angular_speed =
    declare_parameter<double>("max_angular_speed", params_.max_angular_speed);
  params_.max_linear_accel = declare_parameter<double>("max_linear_accel", params_.max_linear_accel);
  params_.max_angular_accel =
    declare_parameter<double>("max_angular_accel", params_.max_angular_accel);
  params_.allow_lateral_motion =
    declare_parameter<bool>("allow_lateral_motion", params_.allow_lateral_motion);
  params_.publish_zero_on_timeout =
    declare_parameter<bool>("publish_zero_on_timeout", params_.publish_zero_on_timeout);
  params_.obstacle_stop_enabled =
    declare_parameter<bool>("obstacle_stop_enabled", params_.obstacle_stop_enabled);
  params_.min_obstacle_distance =
    declare_parameter<double>("min_obstacle_distance", params_.min_obstacle_distance);
  params_.front_arc_degrees = declare_parameter<double>("front_arc_degrees", params_.front_arc_degrees);

  params_.command_timeout = std::max(0.0, params_.command_timeout);
  params_.timer_period = std::max(0.01, params_.timer_period);
  params_.max_linear_speed = std::max(0.0, params_.max_linear_speed);
  params_.max_lateral_speed = std::max(0.0, params_.max_lateral_speed);
  params_.max_angular_speed = std::max(0.0, params_.max_angular_speed);
  params_.max_linear_accel = std::max(0.0, params_.max_linear_accel);
  params_.max_angular_accel = std::max(0.0, params_.max_angular_accel);
  params_.min_obstacle_distance = std::max(0.0, params_.min_obstacle_distance);
  params_.front_arc_degrees = clamp(params_.front_arc_degrees, 0.0, 360.0);
}

void TeleopSafetyNode::handleCommand(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  const auto now = get_clock()->now();
  last_command_time_ = now;
  have_command_ = true;

  publishCommand(makeSafeCommand(*msg, now), now);
}

void TeleopSafetyNode::handleScan(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  const double half_arc_rad = degreesToRadians(params_.front_arc_degrees) * 0.5;
  double nearest = std::numeric_limits<double>::infinity();

  for (std::size_t index = 0; index < msg->ranges.size(); ++index) {
    const double angle = msg->angle_min + (static_cast<double>(index) * msg->angle_increment);
    if (std::abs(angle) > half_arc_rad) {
      continue;
    }

    const double range = msg->ranges[index];
    if (!std::isfinite(range) || range < msg->range_min || range > msg->range_max) {
      continue;
    }

    nearest = std::min(nearest, range);
  }

  nearest_front_obstacle_m_ = nearest;
}

void TeleopSafetyNode::handleEstop(const std_msgs::msg::Bool::SharedPtr msg)
{
  const bool was_active = estop_active_;
  estop_active_ = msg->data;

  if (estop_active_) {
    publishZeroNow();
    if (!was_active) {
      RCLCPP_WARN(get_logger(), "E-stop active; publishing zero velocity");
    }
  } else if (was_active) {
    RCLCPP_INFO(get_logger(), "E-stop cleared");
  }
}

void TeleopSafetyNode::handleSpeedLimit(const std_msgs::msg::Float64::SharedPtr msg)
{
  speed_limit_scale_ = clamp(finiteOrZero(msg->data), 0.0, 1.0);
}

void TeleopSafetyNode::handleWatchdog()
{
  const auto now = get_clock()->now();
  if (commandTimedOut(now) && params_.publish_zero_on_timeout) {
    publishZeroNow();
  }
}

geometry_msgs::msg::Twist TeleopSafetyNode::makeSafeCommand(
  const geometry_msgs::msg::Twist & raw_command,
  const rclcpp::Time & now)
{
  if (estop_active_) {
    return zeroCommand();
  }

  geometry_msgs::msg::Twist safe;
  const double speed_scale = clamp(speed_limit_scale_, 0.0, 1.0);
  const double linear_cap = params_.max_linear_speed * speed_scale;
  const double lateral_cap = params_.max_lateral_speed * speed_scale;
  const double angular_cap = params_.max_angular_speed * speed_scale;

  safe.linear.x = clamp(finiteOrZero(raw_command.linear.x), -linear_cap, linear_cap);
  safe.linear.y = params_.allow_lateral_motion ?
    clamp(finiteOrZero(raw_command.linear.y), -lateral_cap, lateral_cap) : 0.0;
  safe.linear.z = 0.0;
  safe.angular.x = 0.0;
  safe.angular.y = 0.0;
  safe.angular.z = clamp(finiteOrZero(raw_command.angular.z), -angular_cap, angular_cap);

  if (obstacleInStopZone() && safe.linear.x > 0.0) {
    safe.linear.x = 0.0;
  }

  if (have_published_) {
    const double dt = std::max(0.0, (now - last_publish_time_).seconds());
    if (params_.max_linear_accel > 0.0) {
      const double max_linear_delta = params_.max_linear_accel * dt;
      safe.linear.x = rateLimit(safe.linear.x, last_safe_command_.linear.x, max_linear_delta);
      safe.linear.y = rateLimit(safe.linear.y, last_safe_command_.linear.y, max_linear_delta);
    }
    if (params_.max_angular_accel > 0.0) {
      const double max_angular_delta = params_.max_angular_accel * dt;
      safe.angular.z = rateLimit(safe.angular.z, last_safe_command_.angular.z, max_angular_delta);
    }
  }

  return safe;
}

geometry_msgs::msg::Twist TeleopSafetyNode::zeroCommand() const
{
  return geometry_msgs::msg::Twist{};
}

void TeleopSafetyNode::publishCommand(
  const geometry_msgs::msg::Twist & command,
  const rclcpp::Time & now)
{
  command_pub_->publish(command);
  last_safe_command_ = command;
  last_publish_time_ = now;
  have_published_ = true;
}

void TeleopSafetyNode::publishZeroNow()
{
  publishCommand(zeroCommand(), get_clock()->now());
}

bool TeleopSafetyNode::commandTimedOut(const rclcpp::Time & now) const
{
  if (!have_command_) {
    return true;
  }
  return (now - last_command_time_).seconds() > params_.command_timeout;
}

bool TeleopSafetyNode::obstacleInStopZone() const
{
  return params_.obstacle_stop_enabled &&
         nearest_front_obstacle_m_ < params_.min_obstacle_distance;
}

double TeleopSafetyNode::clamp(double value, double min_value, double max_value)
{
  return std::min(std::max(value, min_value), max_value);
}

double TeleopSafetyNode::finiteOrZero(double value)
{
  return std::isfinite(value) ? value : 0.0;
}

double TeleopSafetyNode::rateLimit(double target, double current, double max_delta)
{
  return clamp(target, current - max_delta, current + max_delta);
}

double TeleopSafetyNode::degreesToRadians(double degrees)
{
  constexpr double pi = 3.14159265358979323846;
  return degrees * pi / 180.0;
}

}  // namespace ros2_cpp_teleop_safety_node
