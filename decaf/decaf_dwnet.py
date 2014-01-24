"""
backprop to image layer
"""
import cPickle as pickle
from decaf import base
from decaf.util import translator, transform
import numpy as np
from decaf.tests import unittest_imagenet_pipeline as tpp
from decaf.layers import core_layers
from decaf.layers.cpp import wrapper


import skimage.io
import skimage.transform
import skimage.filter
import scipy.io

DF='/data/vision/billf/stereo-vision/VisionLib/Donglai/DeepL/decaf/'
from collections import namedtuple
FLAGS = namedtuple("FLAGS", "net_file meta_file")
FLAGS.net_file = DF+'scripts/imagenet.decafnet.epoch90'
FLAGS.meta_file = DF+'scripts/imagenet.decafnet.meta'
INPUT_DIM = 227
OUTPUT_DIM = 1000
OUTPUT_AFFIX = '_cudanet_out'
DATA_TYPENAME = 'data'
COST_TYPENAME = 'cost'


class dw_LossLayer(base.LossLayer):
    def set_label(self,label):
        # only care about certain labels
        self.label = label

    def forward(self, bottom, top):
        """Forward emits the loss, and computes the gradient as well."""
        diff = bottom[0].init_diff(setzero=False)
        #scipy.io.savemat('pp.mat',mdict={'im1':bottom[0].data(),'im2':bottom[1].data(),'im3':self.label})
        diff[:] = bottom[0].data()
        diff -= bottom[1].data()
        diff = np.multiply(diff,self.label)
        self._loss = np.dot(diff.flat, diff.flat) / 2. / diff.shape[0] \
                * self.spec['weight']
        diff *= self.spec['weight'] / diff.shape[0]        

class dw_Net(base.Net):
    def data_backward(self,bottom,top,l_conv1):
        """Runs the backward pass."""
        # bottom:(blob) data
        # top:(blob) conv1_neuron
        # l_conv1: (layer) conv1
        top_diff = top[0].diff()
        t_sz = top_diff.shape
        t_sz0 = np.prod(t_sz[:-1])
        top_diff = np.reshape(top_diff,(t_sz0,t_sz[-1]))

        padded_data = l_conv1._padded.data()
        col_data = l_conv1._col.data() # stored in layers['conv1']
        kernel_data = l_conv1._kernels.data()
        k_sz = kernel_data.shape

        col_diff = np.zeros((t_sz0,k_sz[0]),'float32');
        for i in range(k_sz[1]):
            col_diff += np.dot(top_diff[:,i:i+1],kernel_data[:,i:i+1].T)
        
        # accumulate back to image col2im
        padded_diff = np.empty_like(bottom[0].data())
        if l_conv1._large_mem:
            wrapper.im2col_backward(padded_diff, np.reshape(col_diff,(t_sz[0],t_sz[1],t_sz[2],k_sz[0])),
                                    l_conv1._ksize, l_conv1._stride)
        else:
            raise NotImplementedError(str(type(self)) + " does not implement small memory im2col_backward.")
            """
            kernel_diff_buffer = np.empty_like(kernel_diff)
            for i in range(bottom_data.shape[0]):
                # although it is a backward layer, we still need to compute
                # the intermediate results using forward calls.
                wrapper.im2col_forward(padded_data[i:i+1], col_data,
                                       l_conv1._ksize, l_conv1._stride)
                blasdot.dot_firstdims(col_data, top_diff[i],
                                     out=kernel_diff_buffer)
                kernel_diff += kernel_diff_buffer
                if propagate_down:
                    blasdot.dot_lastdim(top_diff[i], l_conv1._kernels.data().T,
                                        out=col_diff)
                    # im2col backward
                    wrapper.im2col_backward(padded_diff[i:i+1], col_diff,
                                            l_conv1._ksize, l_conv1._stride)
            """
        return padded_diff
    def forward_backward(self, previous_net = None):
        """Runs the forward and backward passes of the net.
        """
        # the forward pass. We will also accumulate the loss function.
        if not self._finished:
            raise DecafError('Call finish() before you use the network.')
        if len(self._output_blobs):
            # If the network has output blobs, it usually shouldn't be used
            # to run forward-backward: such blobs won't be used and cause waste
            # of computation. Maybe the user is missing a few loss layers? We
            # will print the warning but still carry on.
            logging.warning('Have multiple unused blobs in the net. Do you'
                            ' actually mean running a forward backward pass?')
        loss = None
        # If there is a previous_net, we will run that first
        if isinstance(previous_net, dw_Net):
            previous_blobs = previous_net.predict()
            try:
                for name in self._input_blobs:
                    self.blobs[name].mirror(previous_blobs[name])
            except KeyError as err:
                raise DecafError('Cannot run forward_backward on a network'
                                 ' with unspecified input blobs.', err)
        elif isinstance(previous_net, dict):
            # If previous net is a dict, simply mirror all the data.
            for key, arr in previous_net.iteritems():
                self.blobs[key].mirror(arr)
        for _, layer, bottom, top in self._forward_order:
            layer.forward(bottom, top)

        # the backward pass
        for name, layer, bottom, top, propagate_down in self._backward_order:
            if layer.name != 'conv1':
                layer.backward(bottom, top, propagate_down)
            else:
                # calculate image gradient
                loss = self.data_backward(bottom,top,layer)
                
        return loss
    def subpatch(self,image,pid=2):
        if pid==1:
            images = np.empty((4, INPUT_DIM, INPUT_DIM, 3), dtype=np.float32)    
            sz = image.shape
            images[0] = image[:INPUT_DIM,:INPUT_DIM]
            images[1] = image[:INPUT_DIM,(sz[1]-INPUT_DIM):]
            images[2] = image[(sz[0]-INPUT_DIM):,:INPUT_DIM]
            images[3] = image[(sz[0]-INPUT_DIM):,(sz[1]-INPUT_DIM):]
        elif pid==2:
            # center patch
            images = np.empty((1, INPUT_DIM, INPUT_DIM, 3), dtype=np.float32)    
            margin = ((image.shape[0]-INPUT_DIM)/2,(image.shape[1]-INPUT_DIM)/2)
            images[0] = image[margin[0]:margin[0]+INPUT_DIM,margin[1]:margin[1]+INPUT_DIM]
        elif pid==3:
            # downsample whole image
            images = np.empty((1, INPUT_DIM, INPUT_DIM, 3), dtype=np.float32)    
            sm = skimage.filter.gaussian_filter(image,sigma = 1/6,multichannel=True)
            images[0] = skimage.transform.resize(sm,(INPUT_DIM,INPUT_DIM))
        return images                                


    def load_jeffnet(self,net_opt):
        self.jf_opt = net_opt
        meta = pickle.load(open(net_opt.meta_file))
        self.jf_label_names = meta['label_names']
        self.jf_data_mean = translator.img_cudaconv_to_decaf(meta['data_mean'], 256, 3)

    def load_data(self,data_opt={'did':0}):
        did = data_opt['did']
        if did==0:
            """We will create a dummy imagenet data of one single image."""
            data = np.random.rand(1, 220, 220, 3).astype(np.float32)
            label = np.zeros((1,1000)).astype(np.float32)
            label[0][np.random.randint(1000, size=1)] = 1
            #label = np.random.randint(1000, size=1)
        elif did==1:
            fns = data_opt['fns']
            pid = data_opt['pid']
            lls = data_opt['lls']
            data = np.empty((0, INPUT_DIM, INPUT_DIM, 3)).astype(np.float32)
            label = np.zeros((0, OUTPUT_DIM)).astype(np.float32)
            for fid,filename in enumerate(fns):
                image = skimage.io.imread(filename).astype(np.uint8)
                image = transform.scale_and_extract(transform.as_rgb(image), 256).astype(np.float32) * 255. - self.jf_data_mean
                data = np.vstack((data,self.subpatch(image,pid))) 
                tmp_ll = np.zeros((data.shape[0]-label.shape[0],1000)).astype(np.float32)
                tmp_ll[:,lls] = 1
                label = np.vstack((label,tmp_ll)) 
                #print data.shape,label.shape        
        self.dataset = core_layers.NdarrayDataLayer(name='input', sources=[data, label])        

    def load_cudanet(self,output_shapes):
        #1. data layer
        self.add_layers(self.dataset,provides=['data', 'label'])
        #2. jeffnet layer
        cuda_layers = pickle.load(open(self.jf_opt.net_file))
        exclude_list = ['probs']
        for cuda_layer in cuda_layers:
            if cuda_layer['name'] not in exclude_list:
                decaf_layer = translator.translate_layer(cuda_layer, output_shapes)
                if not decaf_layer:
                    continue
                needs = []
                for idx in cuda_layer['inputs']:
                    if cuda_layers[idx]['type'] == DATA_TYPENAME:
                        needs.append(cuda_layers[idx]['name'])
                    else:
                        needs.append(cuda_layers[idx]['name'] + OUTPUT_AFFIX)
                provide = cuda_layer['name'] + OUTPUT_AFFIX
                self.add_layers(decaf_layer, needs=needs, provides=provide)
        #3. loss layer
        #loss_layer = core_layers.SquaredLossLayer(name='loss')        
        loss_layer = dw_LossLayer(name='loss')        
        self.add_layer(loss_layer,needs=['fc8_cudanet_out', 'label'])
        self.finish()
        self.layers['loss'].set_label(self.layers['input']._sources[1])

def check_more(net,ll_ran,fn='bp_',num_persave=100):    
    num_left = len(ll_ran)
    num_save = int(np.ceil(float(num_left)/num_persave))
    tmp_sz = net.layers['input']._sources[0].shape
    im = np.zeros((tmp_sz[0],tmp_sz[1],tmp_sz[2],tmp_sz[3],min(num_left,num_persave)),'float32')
    tmp_ll = np.zeros_like(net.layers['input']._sources[1])    
    for s in range(num_save):
        im[:] = 0
        for t in range(num_left):
            i = ll_ran[s*num_persave+t]
            tmp_ll[:,i] = 1
            net.layers['loss'].set_label(tmp_ll)
            im[:,:,:,:,t] = net.forward_backward()            
            tmp_ll[:] = 0
        print "save ",s
        scipy.io.savemat(fn+str(s)+'.mat',mdict={'im':im})
	num_left -= num_persave

def check_one(net):
    im = net.forward_backward()
    scipy.io.savemat('ha.mat',mdict={'im':im})
    scipy.io.savemat('im.mat',mdict={'oo':net.layers['input']._sources[0]})
    a=open('label.txt','w');
    for nn in net.jf_label_names:
        a.write(nn+'\n')
    a.close()

def check_update():
    b = net.layers['fc8']._bias.data()
    bd = net.layers['fc8']._bias.diff.im_self.data()
    net.update()
    bb = net.layers['fc8']._bias.data()

net = dw_Net()
net.load_jeffnet(FLAGS)
tid = 1
if tid==0:
    # baseline: how a x backprop on pubdog (qualitative)
    net.load_data({'did':1,'fns':['test_pugdog.jpg'],'pid':2,'lls':147})
elif tid==1:
    # baseline: how a x backprop on bear (quanatative)
    net.load_data({'did':1,'fns':['100099.jpg'],'pid':2,'lls':147})

net.load_cudanet( {'data': (INPUT_DIM, INPUT_DIM, 3)})
net.layers['conv1']._large_mem = True
import sys
name = int(sys.argv[1])
# name: 0-15
rr = range(name*63,min(1000,(name+1)*63))
check_more(net,rr,'bp'+str(tid)+'_'+str(name)+'_')
"""
for id in {0..15}
do
python decaf_dwnet.py ${id} &
done
"""
