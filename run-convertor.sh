#!/bin/bash
dvs_msg_path=~/catkin_ws
ros_bag_path=/home/eduardo/Documents/ros-bags/Cirs/EBC/marine-snow-Banyoles/11052023/bg_only-tests
ros_bag_file=2023-05-11-19-17-34.bag
event_topic=/dvs/events 
image_topic=/dvs/image_raw

source $dvs_msg_path/devel/setup.bash

python ./events-bag2hdf5.py $ros_bag_path/$ros_bag_file --topic $image_topic $event_topic