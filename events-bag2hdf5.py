#!/usr/bin/env python
import os
import sys

import numpy as np
import argparse
import h5py

import roslib
roslib.load_manifest('rosbag')
import rosbag
import progressbar

import cv2
from cv_bridge import CvBridge

# import the messages of DVS sensor
from dvs_msgs.msg import Event, EventArray
from sensor_msgs.msg import Image

def process_msg(msg,t):
    for i, attr in enumerate(msg.__slots__):
        rostype = msg._slot_types[i]
        #print("\n",rostype," ",i," ",attr,"\n")

        if rostype == 'dvs_msgs/Event[]':
            events = getattr(msg, attr)
            result = np.zeros((len(events),4))
            
            for i in range(len(events)):
                time = events[i].ts.secs + events[i].ts.nsecs*1e-9
                result[i] = np.array([events[i].x,events[i].y,time,events[i].polarity])
#
        elif rostype == 'uint8[]':
            pass

    return result

def bag2hdf5(fname,out_fname,topics=None):
    bag = rosbag.Bag(fname)
    namespace = 'davis/left'
    results2 = {'davis':{'left':{}}}
    chunksize = 10000 # dont know yet what this is
    dsets = {}

    # progressbar
    _pbw = ['converting %s: ' % fname,progressbar.Percentage()]
    pbar = progressbar.ProgressBar(widgets=_pbw, maxval=bag.size).start()

    try:
        with h5py.File(out_fname, mode='w') as out_f:
            for topic, msg, t in bag.read_messages(topics=topics):
                # update progressbar
                #print("\nmsg {0}".format(msg))
                pbar.update(bag._file.tell())

                # convert it to numpy element (and dtype)
                if topic == '/dvs/events':
                     # get the event data
                    this_row = process_msg(msg, t)
                    object = 'events'
                    if object not in results2['davis']['left']:
                        try:
                            dtype = np.ndarray
                        except:
                            print >> sys.stderr, "*********************"
                            print >> sys.stderr, 'topic:', topic
                            print >> sys.stderr, '\nerror while processing message:\n\n%r' % msg
                            print >> sys.stderr, '\nROW:', this_row
                            print >> sys.stderr, "*********************"
                            raise                            
                        results2['davis']['left'][object] = this_row
                    else:
                        results2['davis']['left'][object] = np.append(results2['davis']['left'][object], this_row, axis=0)

                elif topic == '/dvs/image_raw':
                    object = 'image_raw'
                    object2 = 'image_raw_ts'

                    # getting image dimensions from the msg data
                    height = getattr(msg, 'height')
                    width = getattr(msg, 'width')

                    # convert the img msg to numpy array
                    bridge = CvBridge()
                    img = bridge.imgmsg_to_cv2(msg,'mono8')
                    img_3d = np.copy(img)
                    img_3d = img_3d[np.newaxis,:,:] # puts the image in a 3 dimensional array for indexing individual images later

                    # Getting timestamp data
                    t = getattr(msg, 'header').stamp
                    time = np.array([t.secs + t.nsecs*1e-9])

                    # Saving img data
                    if object not in results2['davis']['left']:
                        try:
                            dtype = np.ndarray
                        except:
                            print >> sys.stderr, "*********************"
                            print >> sys.stderr, 'topic:', topic
                            print >> sys.stderr, '\nerror while processing message:\n\n%r' % msg
                            print >> sys.stderr, '\nROW:', img_3d
                            print >> sys.stderr, "*********************"
                            raise                            
                        results2['davis']['left'][object] = img_3d
                    else:
                        results2['davis']['left'][object] = np.append(results2['davis']['left'][object], img_3d, axis=0)
                    
                    # Saving img timestamp data
                    if object2 not in results2['davis']['left']:
                        try:
                            dtype = np.ndarray
                        except:
                            print >> sys.stderr, "*********************"
                            print >> sys.stderr, 'topic:', topic
                            print >> sys.stderr, '\nerror while processing message:\n\n%r' % msg
                            print >> sys.stderr, '\nROW:', time
                            print >> sys.stderr, "*********************"
                            raise                            
                        results2['davis']['left'][object2] = time
                    else:
                        results2['davis']['left'][object2] = np.append(results2['davis']['left'][object2], time)
                
                # flush the caches periodically
                if len(results2['davis']['left'][object]) >= chunksize:
                    if object not in dsets:
                        # initial creation
                        if object is 'image_raw':
                            # img dataset
                            dset = out_f.create_dataset(namespace+'/'+object, data=results2['davis']['left'][object],
                                                        maxshape=(None,height,width),
                                                        compression='gzip',
                                                        compression_opts=9)
                            assert dset.compression == 'gzip'
                            assert dset.compression_opts == 9
                            dsets[object] = dset

                            # timestamp dataset
                            dset2 = out_f.create_dataset(namespace+'/'+object2, data=results2['davis']['left'][object2],
                                                        maxshape=(None,),
                                                        compression='gzip',
                                                        compression_opts=9)
                            assert dset.compression == 'gzip'
                            assert dset.compression_opts == 9
                            dsets[object2] = dset2
                        elif object is 'events':
                            dset = out_f.create_dataset(namespace+'/'+object, data=results2['davis']['left'][object],
                                                        maxshape=(None,4),
                                                        compression='gzip',
                                                        compression_opts=9)
                            assert dset.compression == 'gzip'
                            assert dset.compression_opts == 9
                            dsets[object] = dset
                        
                    else:
                        # append to existing dataset
                        if object is 'image_raw':
                            h5append(dsets[object], results2['davis']['left'][object])
                            h5append(dsets[object2], results2['davis']['left'][object2])
                        elif object is 'events':
                            h5append(dsets[object], results2['davis']['left'][object])
                    # clear the cache values
                    if object is 'events':
                        results2['davis']['left'][object] = np.empty((0,4))
                    if object is 'image_raw':
                        # cleaning img cache
                        results2['davis']['left'][object] = np.empty((0,height,width))
                        results2['davis']['left'][object2] = np.empty((0,1))
                    else:
                        pass
            
            # done reading bag file. flush remaining data to h5 file
            for object in results2['davis']['left']:
                if not len(results2['davis']['left'][object]):
                    # no data
                    continue
                if object in dsets:
                    if object is 'image_raw':
                        h5append(dsets[object], results2['davis']['left'][object])
                        h5append(dsets[object2], results2['davis']['left'][object2])
                    elif object is 'events':
                        h5append(dsets[object], results2['davis']['left'][object])
                else:
                    if object is 'image_raw':
                        # img dataset
                        dset = out_f.create_dataset(namespace+'/'+object, data=results2['davis']['left'][object],
                                                    maxshape=(None,height,width),
                                                    compression='gzip',
                                                    compression_opts=9)
                        assert dset.compression == 'gzip'
                        assert dset.compression_opts == 9
                        dsets[object] = dset

                        # timestamp dataset
                        dset2 = out_f.create_dataset(namespace+'/'+object2, data=results2['davis']['left'][object2],
                                                    maxshape=(None,),
                                                    compression='gzip',
                                                    compression_opts=9)
                        assert dset.compression == 'gzip'
                        assert dset.compression_opts == 9
                        dsets[object2] = dset2
                        #print '\nD',dsets['image_raw_ts'].shape
                        #try:
                        #    input("Press enter to continue")
                        #except SyntaxError:
                        #    pass
                    elif object is 'events':
                        dset = out_f.create_dataset(namespace+'/'+object, data=results2['davis']['left'][object],
                                                    maxshape=(None,4),
                                                    compression='gzip',
                                                    compression_opts=9)
                        assert dset.compression == 'gzip'
                        assert dset.compression_opts == 9
                        dsets[object] = dset
                    
    except:
        if os.path.exists(out_fname):
            os.unlink(out_fname)
        raise
    finally:
        pbar.finish()

def make_dtype(msg):
    result = []
    for i, attr in enumerate(msg._slot_types):
        rostype = msg._slot_types[i]

        if rostype == 'dvs_msgs/Event[]':
            result.extend([('events_x',np.uint16),
                           ('events_y',np.uint16),
                           ('events_ts',np.uint64),
                           ('events_p',np.bool_)])
    
    return result

def h5append(dset,arr):
    n_old_rows = dset.shape[0]
    n_new_rows = len(arr) + n_old_rows
    dset.resize(n_new_rows, axis=0)
    dset[n_old_rows:] = arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts event data from rosbag file to hdf5 format')
    parser.add_argument('filename', type=str, help="the .bag file")
    parser.add_argument('--out', type=str, default=None, help="name of output file.")
    parser.add_argument('--topic', type=str, nargs='*', help="topic name to convert. Defaults to all. Multiple can be specified.")

    args = parser.parse_args()

    # check if the provided file exists
    if not os.path.exists(args.filename):
        print >> sys.stderr, "No file %s" %args.filename
        sys.exit(1)

    # saving the file 
    fname = os.path.splitext(args.filename)[0]
    if args.out is not None:
        output_fname = args.out
    else:
        output_fname = fname + '.hdf5'
        if os.path.exists(output_fname):
            print >> sys.stderr, 'will not overwrite %s.' % output_fname
            sys.exit(1)

    bag2hdf5(args.filename, output_fname, topics = args.topic)