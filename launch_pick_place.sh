#!/bin/bash

# Настройка окружения ROS 2
source /opt/ros/humble/setup.bash

# Настройка твоего workspace
source ~/damir_ros/install/setup.bash

# Запуск launch-файла
gnome-terminal -- bash -c "ros2 run dobot_sorting_color main_new; exec bash"
sleep 1
gnome-terminal -- bash -c "ros2 run dobot_sorting_color camera_new; exec bash"

echo "Dobot Pick and Place Launched Successfully"
