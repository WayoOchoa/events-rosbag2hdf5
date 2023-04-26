# Python events-rosbag2hdf5
Converts a rosbag file containing event camera data into hdf5 standard format

This python code converts a rosbag file into HDF5 data file. The rosbag should have been recorded using the [rpg_dvs_ros](https://github.com/uzh-rpg/rpg_dvs_ros) package or have the event and image data topics structured the same as the mentioned package.

The data in the HDF5 file is structured in the same way as the MVSEC dataset,e.g., to access image data in your code:
        `f = h5py.File(file_name, mode='r')`
        `img_data = f['davis']['left']['image_raw']`

## Run
To run the python script, use the *run-convertor.sh* bash file that is in the directory. This file contains variables that should be modified to specify your catkin_ws workspace path, the rosbag path and name for the bag that you will convert, and the name of the topics for the events and image data.
Run the script by making it executable and then in a command window:
        `./run-convertor.sh`