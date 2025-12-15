# Humanoid Robot Simulation in ROS 2

This project provides a framework for simulating a humanoid robot using ROS 2. It includes configurations for joint names, launch files for visualization and simulation, and the necessary URDF model for the robot.

## Project Structure

- **config/joint_names_Humanoid_10102025.yaml**: Contains the configuration for joint names used in the humanoid robot, mapping joint names to their respective identifiers.

- **launch/display.launch.py**: Launch file responsible for starting display nodes for visualizing the robot in a simulation environment, including configurations for RViz or other visualization tools.

- **launch/gazebo.launch.py**: Launch file that sets up the Gazebo simulation environment for the humanoid robot, including parameters for loading the robot model and necessary plugins.

- **meshes/**: Directory intended to contain 3D mesh files for the robot's visual representation, including files in formats such as STL or COLLADA.

- **package.xml**: Defines the package metadata for the ROS 2 package, including package name, version, maintainers, dependencies, and licensing.

- **README.md**: Documentation for the project, including setup instructions, usage guidelines, and relevant information about the humanoid robot.

- **src/humanoid_state_publisher.cpp**: C++ source file implementing a node that publishes the state of the humanoid robot's joints, subscribing to joint state messages and publishing them to the appropriate topics.

- **urdf/Humanoid_10102025.urdf.xacro**: XACRO file that generates the URDF representation of the humanoid robot, allowing for parameterization and modularization of the robot model.

- **worlds/humanoid_world.sdf**: Defines the simulation world in which the humanoid robot will operate, specifying the environment, including obstacles, ground planes, and other elements.

- **CMakeLists.txt**: Used for building the ROS 2 package, specifying build instructions, dependencies, and targets for the package.

## Setup Instructions

1. **Install ROS 2**: Follow the official ROS 2 installation guide for your operating system.

2. **Clone the Repository**: Clone this repository to your local machine.

3. **Build the Package**: Navigate to the root of the project and run:
   ```
   colcon build
   ```

4. **Source the Setup File**: After building, source the setup file:
   ```
   source install/setup.bash
   ```

5. **Launch the Simulation**: Use the provided launch files to start the simulation:
   ```
   ros2 launch humanoid_10102025_ros2 gazebo.launch.py
   ```

## Usage Guidelines

- Use the `display.launch.py` file to visualize the robot in RViz.
- Modify the `joint_names_Humanoid_10102025.yaml` file to customize joint configurations.
- Add any additional mesh files to the `meshes` directory for enhanced visual representation.

This project is structured to support the development and simulation of a humanoid robot using ROS 2, with configurations for visualization, simulation, and robot state management.