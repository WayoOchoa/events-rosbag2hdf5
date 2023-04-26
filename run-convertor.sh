#!/bin/bash
dvs_msg_path=~/catkin_ws
ros_bag_path=/home/eduardo/Documents/ros-bags/Cirs/EBC/Sant-Feliu_MarineSnowTesting_14042023
ros_bag_file=session1_space-surface_marine-snow-wall_14042023-test7.bag
event_topic=/dvs/events 
image_topic=/dvs/image_raw

source $dvs_msg_path/devel/setup.bash

python ./events-bag2hdf5.py $ros_bag_path/$ros_bag_file --topic $image_topic $event_topic