# events-rosbag2hdf5
Converts a rosbag file containing event camera data into hdf5 standard format

This python code converts a rosbag file into HDF5 data file. The rosbag should have been recorded using the rpg_dvs_ros package or have the event
and image data topics structured the same as the mentioned package.

The data in the HDF5 file is structured in the same way as the MVSEC dataset,e.g., to access image data when loading use:
                        data['davis']['left']['image_raw']
