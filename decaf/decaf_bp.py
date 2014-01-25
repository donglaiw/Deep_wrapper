"""The main routine that starts a imagenet demo."""
from decaf.scripts import imagenet
import datetime
import glob
import gflags
import logging
import numpy as np
import os
from PIL import Image as PILImage
import skimage.io
import skimage.transform
import skimage.filter
import cStringIO as StringIO
import sys
import time
import urllib
from werkzeug import secure_filename

import cPickle as pickle
INPUT_DIM = 227
INPUT_DIM_1 = 55
"""
gflags.DEFINE_string('net_file', '', 'The network file learned from cudaconv')
gflags.DEFINE_string('meta_file', '', 'The meta file for imagenet.')
FLAGS = gflags.FLAGS
"""
from collections import namedtuple
FLAGS = namedtuple("FLAGS", "net_file meta_file")
FLAGS.net_file = '../scripts/imagenet.decafnet.epoch90'
FLAGS.meta_file = '../scripts/imagenet.decafnet.meta'

from decaf.util import translator, transform    
_JEFFNET_FLIP = True

if __name__ == '__main__':    
    net = imagenet.DecafNet(net_file=FLAGS.net_file, meta_file=FLAGS.meta_file)
    
    import scipy.io  
    meta = pickle.load(open(FLAGS.meta_file))
    data_mean = translator.img_cudaconv_to_decaf(meta['data_mean'], 256, 3)
    
    TT = int(sys.argv[1]) 
    if TT==-1:
        fns = ['./test.jpg']
    else:
        if TT==0:
            DD= '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/images/train/'
        if TT==1:
            DD= '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/images/val/'  
        elif TT==2:
            DD= '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/images/test/'  
        fns = sorted(glob.glob(DD+'*.jpg'))

    patch_id = 3
    if patch_id==1:
        num_patch= 4
    elif patch_id ==2:
        num_patch= 1
    elif patch_id ==3:
        num_patch= 1

    num_monitor = 10
    do_pred = 1
    for ii,filename in enumerate(fns):
        nn = filename[filename.rfind('/')+1:-4]
        image = skimage.io.imread(filename).astype(np.uint8)
        images = subpatch(image,patch_id)
        scores = net.classify_direct(images)
        if do_pred:
            scores =scores.mean(0)
            indices, predictions = net.top_k_prediction(scores, 5)
            for i, p in zip(indices, predictions):
                print p,' %.5f' % scores[i]
        
        #break
        if ii%num_monitor==num_monitor-1:
            print ii
    #a.close()
    #scipy.io.savemat('decaf_'+str(patch_id)+'_'+str(TT)+'.mat',mdict={'conv1':conv[0],'conv2':conv[1],'conv3':conv[2],'conv4':conv[3],'conv5':conv[4]})       

