import sys
import os
import numpy as np
import tensorflow as tf
import datetime
import Hill_Climbing
import Simulated_annealing
from data_handler import Data_handler

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('num_iter', 10000, 'Total training iterations')
flags.DEFINE_string('model_dir', 'C:\\Users\\Mohammad\\PycharmProjects\\Final_2d_stereo_matching\\NCS2_randomforest\\try3', 'Trained network dir')
flags.DEFINE_string('data_version', 'kitti2015', 'kitti2012 or kitti2015')
flags.DEFINE_string('data_root', 'E:\\training', 'training dataset dir')
flags.DEFINE_string('util_root', 'E:\\debug_15', 'Binary training files dir')
flags.DEFINE_string('net_type', 'win37_dep9', 'Network type: win37_dep9 pr win19_dep9')

flags.DEFINE_integer('eval_size', 200, 'number of evaluation patchs per iteration')
flags.DEFINE_integer('num_tr_img', 160, 'number of training images')
flags.DEFINE_integer('num_val_img', 40, 'number of evaluation images')
flags.DEFINE_integer('patch_size', 37, 'training patch size')
flags.DEFINE_integer('num_val_loc', 50000, 'number of validation locations')
flags.DEFINE_integer('disp_range', 201, 'disparity range')
flags.DEFINE_string('phase', 'train', 'train or evaluate')

FLAGS = flags.FLAGS

np.random.seed(123)

dhandler = Data_handler(data_version=FLAGS.data_version,
                        data_root=FLAGS.data_root,
                        util_root=FLAGS.util_root,
                        num_tr_img=FLAGS.num_tr_img,
                        num_val_img=FLAGS.num_val_img,
                        num_val_loc=FLAGS.num_val_loc,
                        batch_size=FLAGS.batch_size,
                        patch_size=FLAGS.patch_size,
                        disp_range=FLAGS.disp_range)

if FLAGS.data_version == 'kitti2012':
    num_channels = 1
elif FLAGS.data_version == 'kitti2015':
    num_channels = 3
else:
    sys.exit('data_version should be either kitti2012 or kitti2015')

class DATA:
    def __init__(self,timefile,batch_size,num_it,path,data_root,util_root,num_val_loc):
        self.batch_size=batch_size
        self.num_iter=num_it
        self.path=path
        self.data_root=data_root
        self.util_root=util_root
        self.num_val_loc=num_val_loc
        self.dhandler=dhandler
        self.timefile=timefile
if __name__=="__main__":
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    timefile = os.path.join(path_to_script, "time_file.txt")
    with open(timefile, "w") as handle:
        print('start time=\t'+str(datetime.datetime.now())+'\n', file=handle)
    data=DATA(timefile,FLAGS.batch_size,FLAGS.num_iter,FLAGS.model_dir,FLAGS.data_root,FLAGS.util_root,FLAGS.num_val_loc)
    #Initialize Process by random Model
    init=[[['conv2d', 64, 'valid', 3], ['none', 0, 'none', 0], ['conv2d', 32, 'valid', 7], ['conv2d', 32, 'same', 11], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['conv2d', 32, 'same', 7], ['conv2d', 32, 'same', 3], ['conv2d', 32, 'same', 5], ['batch', 0, 'none', 0], ['none', 0, 'none', 0], ['conv2d', 32, 'valid', 29], ['batch', 0, 'none', 0]], [['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['conv2d', 32, 'valid', 37]]]


    '''[[['conv2d', 32, 'same', 5],
           ['conv2d', 64, 'same', 5],
           ['none', 0, 'none', 0],
           ['conv2d', 64, 'same', 5],
           ['none', 0, 'none', 0],
           ['conv2d', 64, 'same', 5],
           ['conv2d', 64, 'same', 5],
           ['none', 0, 'none', 0],
           ['conv2d', 64, 'valid', 3],
           ['conv2d', 64, 'same', 5],
           ['conv2d', 64, 'same', 5],
           ['conv2d', 32, 'valid', 35]],
          [['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['conv2d', 32, 'valid', 37]]]'''
    #init=[[['none', 0, 'none', 0], ['conv2d', 64, 'valid', 3], ['conv2d', 32, 'valid', 7], ['conv2d', 32, 'same', 5], ['conv2d', 32, 'same', 11], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['conv2d', 32, 'same', 7], ['conv2d', 32, 'same', 5], ['conv2d', 32, 'same', 3], ['batch', 0, 'none', 0], ['none', 0, 'none', 0], ['conv2d', 32, 'valid', 29]], [['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['conv2d', 32, 'valid', 37]]]


    #Set Idle Fraction Rate for Hill Climbing
    idle_fraction=0.02
    #Start Late Acceptance Hill Climbing Optimization Process by Initial Random Model
    HC_state=Hill_Climbing.Hill_Climber(data,idle_fraction,init)
    #Start Simulated Annealing by LAHC's Best Model
    SA_state=Simulated_annealing.Sim_Annealer(data,HC_state)




