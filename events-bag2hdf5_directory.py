#!/usr/bin/env python
"""
Script to convert event data from ROS bag files to HDF5 format.

This script reads event and image messages from a ROS bag file, processes the messages
in parallel using multiprocessing, and stores the data in HDF5 format. It extracts event
data and image data from the messages and associates the images with the closest event
in time. The processed data is then saved to an HDF5 file.

Author(s): Eduardo Ochoa, Moses Chuka Ebere

Usage:
    python convert_rosbag_to_hdf5.py --directory <input_directory> --out <output_directory> --topic <topic_name> ...

Parameters:
    --directory (str): Path to the directory containing the ROS bag files to convert.
    --out (str, optional): Path to the output directory where the HDF5 files will be saved. Default is the same as input directory.
    --topic (str, optional): Name of the topic to convert. Defaults to all topics. Multiple topics can be specified.

Dependencies:
    - numpy
    - argparse
    - h5py
    - roslib
    - rosbag
    - multiprocessing
    - tqdm
    - typing
    - pathlib
    - natsort
    - time
    - cv_bridge

Functions:
    process_msg: Extracts events from a ROS message and returns them as a NumPy array.
    process_image_and_time: Extracts image data and timestamps from a ROS message and returns them as NumPy arrays.
    bag2hdf5: Converts event data from ROS bag file to HDF5 format and saves the data to an output file.
    closest_event_to_image: Associates images with the closest event generated in time.

"""

import os
import sys

import numpy as np
import argparse
import h5py

import roslib
roslib.load_manifest('rosbag')
import rosbag
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Tuple

import pathlib
import natsort

import time
from cv_bridge import CvBridge


def process_msg(msg) -> np.ndarray:
    """
    Extracts events from a ROS message and returns them as a NumPy array.

    Args:
        msg (rosbag.Message): The ROS message containing events.

    Returns:
        np.ndarray: A NumPy array containing the extracted events. The shape of the array is (N, 4), where N is the number of events.
            The array has four columns: [x_position, y_position, timestamp, polarity].
    """
    # Extract all events contained in the message
    events = msg.events
    # Extract all events' data into separate NumPy arrays
    times = np.array([event.ts.secs + event.ts.nsecs * 1e-9 for event in events])
    # Convention for polarity: +1/-1
    polarities = np.array([1 if event.polarity else -1 for event in events])
    positions = np.array([[event.x, event.y] for event in events])
    
    # Combine the extracted arrays to form the result array
    result = np.column_stack((positions, times, polarities))

    return result

def process_image_and_time(msg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts image data and timestamps from a ROS message and returns them as NumPy arrays.

    Args:
        msg (rosbag.Message): The ROS message containing the image data and timestamp.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - The first element of the tuple is a 3D NumPy array representing the image data. The shape of the array is (1, height, width),
              where height and width are the dimensions of the image.
            - The second element of the tuple is a 1D NumPy array representing the timestamp of the image.
    """
    # Getting timestamp data
    t = getattr(msg, 'header').stamp
    time_ = np.array([t.secs + t.nsecs*1e-9])

    # convert the img msg to numpy array
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg,'mono8')
    img_3d = np.copy(img)
    img_3d = img_3d[np.newaxis,:,:] # puts the image in a 3 dimensional array for indexing individual images later
    if msg.width != 346:
        return np.full((1, 260, 346), -1, dtype=np.int8), time_
    return img_3d, time_

def bag2hdf5(fname: pathlib.PosixPath, out_fname: pathlib.PosixPath,topics=None):
    """
    Converts event data from ROS bag file to HDF5 format and saves the data to an output file.

    Args:
        fname (pathlib.PosixPath): Path to the input ROS bag file.
        out_fname (pathlib.PosixPath): Path to the output HDF5 file.
        topics (List[str], optional): List of topic names to convert. Defaults to None, which means converting all topics.

    Returns:
        None: This function does not return any value. It saves the processed data to the HDF5 file specified by out_fname.
    """
    # Extract the bag file
    bag = rosbag.Bag(fname)
    event_dict = {}

    # Extract the event messages and image messages from the bag file. 
    if topics[0] == '/dvs/events':
        events_msgs = [msg for _, msg, _ in bag.read_messages(topics=topics[0])]
        image_msgs = [msg for _, msg, _ in bag.read_messages(topics=topics[1])]
    else:
        events_msgs = [msg for _, msg, _ in bag.read_messages(topics=topics[1])]
        image_msgs = [msg for _, msg, _ in bag.read_messages(topics=topics[0])]

    height, width = 260, 346 # TODO: Place these in a config file. 
        
    ## Process Events
    # Number of processes to use (you can adjust this based on your CPU core count)
    num_processes = int(cpu_count()/2)

    # Create a pool of worker processes
    pool = Pool(processes=num_processes)
    # print(type(all_msgs['/dvs/events']))
    # Use the pool.map to parallelize the process_msg function
    event_data = pool.map(process_msg, events_msgs)
    # Close the pool to release resources
    pool.close()

    # Combine all results into a single large matrix
    event_data = np.vstack(event_data)

    ## Process Images and Time Stamps
    # Number of processes to use (you can adjust this based on your CPU core count)
    num_processes = int(cpu_count()/2)

    # Create a pool of worker processes
    pool1 = Pool(processes=num_processes)
    # Use the pool.map to parallelize the process_msg function
    output_matrix = pool1.map(process_image_and_time, image_msgs)
    # Unpack the results into separate arrays
    image_arrays, time_arrays = zip(*output_matrix)
    # Convert the lists of arrays into NumPy arrays if needed
    image_arrays = np.vstack(image_arrays)
    time_arrays = np.vstack(time_arrays).reshape(-1,)
    # Close the pool to release resources
    pool1.close()

    # Extract the event indices based the closest event to an image
    event_ids = closest_event_to_image(event_data, time_arrays)


    # Open a new hdf5 file and write to it
    with h5py.File(out_fname, mode='w') as out_f:
        # Create the 'davis' group
        davis_group = out_f.create_group('davis')

        # Create the 'left' group within the 'davis' group
        left_group = davis_group.create_group('left')

        # Add the dataset 'events' to the 'left' group
        left_group.create_dataset('events', data=event_data, 
                                    maxshape=(None, 4), compression='gzip', compression_opts=9)
        left_group.create_dataset('image_raw', data=image_arrays, 
                                    maxshape=(None, height, width), compression='gzip', compression_opts=9)
        left_group.create_dataset('image_raw_ts', data=time_arrays, 
                                    maxshape=(None,), compression='gzip', compression_opts=9)
        left_group.create_dataset('image_raw_event_inds', data=event_ids, 
                                    maxshape=(None, 1), compression='gzip', compression_opts=9)


def closest_event_to_image(dset_events: np.ndarray, dset_img_timestamps: np.ndarray, batch_size: int=80) -> np.ndarray:
    """
    Associates images with the closest event generated in time.

    Args:
        dset_events (np.ndarray): NumPy array containing the event data. The shape of the array is (N, 4), where N is the number of events.
            The array has four columns: [x_position, y_position, timestamp, polarity].
        dset_img_timestamps (np.ndarray): NumPy array containing the timestamps of the images. The shape of the array is (M,),
            where M is the number of images.
        batch_size (int, optional): Batch size for processing the data in batches. Defaults to 80.

    Returns:
        np.ndarray: A NumPy array containing the indices of the closest events to each image timestamp.
            The shape of the array is (M, 1), where M is the number of images.
    """

    # Get the total number of data points
    total_points = dset_img_timestamps.shape[0]

    # Create an empty array to store the results
    closest_event_ids = np.zeros((total_points, 1))

    # Convert h5py datasets to NumPy arrays
    dset_events = dset_events[:]
    dset_img_timestamps = dset_img_timestamps[:]

    # Preallocate the differences array
    differences = np.zeros((dset_events.shape[0], batch_size))

    # Process the data in batches
    for i in range(0, total_points, batch_size):
        # Calculate the range for the current batch
        start_idx = i
        end_idx = min(i + batch_size, total_points)

        # Calculate the absolute differences between all image timestamps and event timestamps for the batch
        differences = np.abs(dset_events[:, 2] - dset_img_timestamps[start_idx:end_idx, np.newaxis])

        # Find the index of the minimum difference for each image timestamp in the batch
        batch_closest_event_ids = np.argmin(differences, axis=1)

        # Store the results in the appropriate indices of the result array
        closest_event_ids[start_idx:end_idx, 0] = batch_closest_event_ids
    return closest_event_ids

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Converts event data from rosbag file to hdf5 format')
    parser.add_argument('--directory', type=str, required=True, help="the directory for the bagfiles")
    parser.add_argument('--out', type=str, default=None, help="name of output directory.")
    parser.add_argument('--topic', type=str, nargs='*', help="topic name to convert. Defaults to all. Multiple can be specified.")

    args = parser.parse_args()

    # Set the input and output directories
    input_dir = pathlib.Path(args.directory)
    assert input_dir.exists(), "Input directory does not exist"

    if args.out is not None:
        output_dir = pathlib.Path(args.out)
        os.makedirs(output_dir, exist_ok=True)
        assert output_dir.exists(), "Output directory does not exist"

    # Use tqdm to show progress bar for the conversion process
    # Sort the bag files in the input directory
    input_files = natsort.natsorted(input_dir.glob("*.bag"))
    # Iterate over the files in the input directory and convert them to hdf5 format.
    for file in tqdm(input_files):
        # check if the provided file exists
        if not os.path.exists(file):
            print("No file %s" %args.directory)
            sys.exit(1)
        # saving the file 
        fname = file.stem
        if args.out is not None:
            output_fname = output_dir / (fname + '.hdf5')
        else:
            output_fname = input_dir / (fname + '.hdf5')
            if os.path.exists(output_fname):
                print('will not overwrite %s.' % output_fname)
                sys.exit(1)
        bag2hdf5(file, output_fname, topics = args.topic)
    end_time = time.time()
    print(end_time - start_time, " -----------------------------")