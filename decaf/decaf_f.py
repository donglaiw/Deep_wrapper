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

def doCMU():
    #gflags.FLAGS(sys.argv)
    logging.getLogger().setLevel(logging.INFO)    
    net = imagenet.DecafNet(net_file=FLAGS.net_file, meta_file=FLAGS.meta_file)
    DD = '/home/Stephen/Desktop/Data/Occ/CMU/ref_frames/'
    fns = glob.glob(DD+'*.tif')
    a=open('vv_decaf.html','w')
    a.write('<table border=1>\n')
    for filename in fns[:1]:
        image = skimage.io.imread(filename).astype(np.uint8)
        scores = net.classify(image)

        indices, predictions = net.top_k_prediction(scores, 5)
        # In addition to the prediction text, we will also produce the length
        # for the progress bar visualization.
        max_score = scores[indices[0]]
        meta = [(p, '%.5f' % scores[i]) for i, p in zip(indices, predictions)]
        logging.info('result: %s', str(meta))
        nn = 'CMU_'+filename[filename.rfind('/')+1:filename.rfind('.')]+'_gtim.png'
        a.write('<tr><td rowspan=5><img width=200 src="orig_im/'+nn+'"></td>\n')
        for i in range(5):
            a.write('<td>'+str(meta[i][0])+'</td><td>'+str(meta[i][1])+'</td></tr>\n<tr>')
        a.write('</tr>\n')
        #break
    a.write('</table>')
    a.close()

from decaf.util import translator, transform    
_JEFFNET_FLIP = True

def net_input(image, data_mean, center_only=False):    
    image = transform.scale_and_extract(transform.as_rgb(image), 256)
    # convert to [0,255] float32
    image = image.astype(np.float32) * 255.
    if _JEFFNET_FLIP:
        # Flip the image if necessary, maintaining the c_contiguous order
        image = image[::-1, :].copy()
    # subtract the mean
    image -= data_mean
    # oversample the images
    images = imagenet.DecafNet.oversample(image, center_only)
    return images

def subpatch(image,id=1):
    if id==1:
        images = np.empty((4, INPUT_DIM, INPUT_DIM, 3), dtype=np.float32)    
        sz = image.shape
        images[0] = image[:INPUT_DIM,:INPUT_DIM]
        images[1] = image[:INPUT_DIM,(sz[1]-INPUT_DIM):]
        images[2] = image[(sz[0]-INPUT_DIM):,:INPUT_DIM]
        images[3] = image[(sz[0]-INPUT_DIM):,(sz[1]-INPUT_DIM):]
    elif id==2:
        images = np.empty((1, INPUT_DIM, INPUT_DIM, 3), dtype=np.float32)    
        margin = ((image.shape[0]-INPUT_DIM)/2,(image.shape[1]-INPUT_DIM)/2)
        #print margin
        images[0] = image[margin[0]:margin[0]+INPUT_DIM,margin[1]:margin[1]+INPUT_DIM]
    elif id==3:
        images = np.empty((1, INPUT_DIM, INPUT_DIM, 3), dtype=np.float32)    
        sm = skimage.filter.gaussian_filter(image,sigma = 1/6,multichannel=True)
        images[0] = skimage.transform.resize(sm,(INPUT_DIM,INPUT_DIM))

    return images                                


    return images
if __name__ == '__main__':    
    net = imagenet.DecafNet(net_file=FLAGS.net_file, meta_file=FLAGS.meta_file)
    
    import scipy.io  
    meta = pickle.load(open(FLAGS.meta_file))
    data_mean = translator.img_cudaconv_to_decaf(meta['data_mean'], 256, 3)
    
    TT = int(sys.argv[1]) 
    if TT==0:
        DD= '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/images/train/'
    if TT==1:
        DD= '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/images/val/'  
    elif TT==2:
        DD= '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/images/test/'  

    #a= open('decaf_score_'+str(TT)+'.txt','w')
    #DD = 'test/'
    fns = sorted(glob.glob(DD+'*.jpg'))
    #print fns
    #sys.exit(0)
    ll= ['conv','_neuron_cudanet_out']
    patch_id = 3
    if patch_id==1:
        num_patch= 4
    elif patch_id ==2:
        num_patch= 1
    elif patch_id ==3:
        num_patch= 1

    num_monitor = 10

    conv = [np.empty((len(fns),num_patch, 55,55, 96), dtype=np.float32), 
             np.empty((len(fns),num_patch, 27,27, 256), dtype=np.float32),        
             np.empty((len(fns),num_patch, 13,13, 384), dtype=np.float32),        
             np.empty((len(fns),num_patch, 13,13, 384), dtype=np.float32),        
             np.empty((len(fns),num_patch, 13,13, 256), dtype=np.float32)]        
    for ii,filename in enumerate(fns):
        nn = filename[filename.rfind('/')+1:-4]
        #print nn
        #print i,nn;sys.exit(0)

        image = skimage.io.imread(filename).astype(np.uint8)
        #im = net_input(image,data_mean,True)    
        #scipy.io.savemat(nn+'.mat',mdict={'im':im})
        images = subpatch(image,patch_id)

        scores = net.classify_direct(images)
        """
        scores =scores.mean(0)
        indices, predictions = net.top_k_prediction(scores, 5)
        a.write(nn+' ')
        for i, p in zip(indices, predictions):
            a.write(p+' %.5f,' % scores[i])
        a.write('\n')
        #mat = net._net.feature(ll)
        #scipy.io.savemat(nn+ll[:ll.find('_')]+'.mat',mdict={'mat':mat})               
        mats[ii] = net._net.feature(ll) 
        """
        for j in range(5):
            conv[j][ii] = net._net.feature(ll[0]+str(j+1)+ll[1]) 
        #break
        if ii%num_monitor==num_monitor-1:
            print ii
    #a.close()
    scipy.io.savemat('decaf_'+str(patch_id)+'_'+str(TT)+'.mat',mdict={'conv1':conv[0],'conv2':conv[1],'conv3':conv[2],'conv4':conv[3],'conv5':conv[4]})       

    """
    for ll in net._net.blobs:
        if 'neuron_cudanet_out' in ll:
            mat = net._net.feature(ll) 
            scipy.io.savemat(nn+ll[:ll.find('_')]+'.mat',mdict={'mat':mat})       

    """
