#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "yaml-cpp/yaml.h"

class HumanoidStatePublisher : public rclcpp::Node
{
public:
    HumanoidStatePublisher()
        : Node("humanoid_state_publisher")
    {
        // Load joint names from the configuration file
        load_joint_names();

        // Create a publisher for joint states
        joint_state_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);

        // Timer to publish joint states at a fixed rate
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&HumanoidStatePublisher::publish_joint_states, this));
    }

private:
    void load_joint_names()
    {
        YAML::Node config = YAML::LoadFile("config/joint_names_Humanoid_10102025.yaml");
        for (const auto& joint : config)
        {
            joint_names_.push_back(joint.as<std::string>());
        }
    }

    void publish_joint_states()
    {
        auto message = sensor_msgs::msg::JointState();
        message.header.stamp = this->get_clock()->now();
        message.name = joint_names_;
        message.position.resize(joint_names_.size(), 0.0); // Placeholder for joint positions

        joint_state_publisher_->publish(message);
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<std::string> joint_names_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HumanoidStatePublisher>());
    rclcpp::shutdown();
    return 0;
}