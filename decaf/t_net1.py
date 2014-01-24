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


from collections import namedtuple
FLAGS = namedtuple("FLAGS", "net_file meta_file")
FLAGS.net_file = '../scripts/imagenet.decafnet.epoch90'
FLAGS.meta_file = '../scripts/imagenet.decafnet.meta'
INPUT_DIM = 227
OUTPUT_AFFIX = '_cudanet_out'
# DATA_TYPENAME is the typename for the data layers at cuda convnet.
DATA_TYPENAME = 'data'
# likewise, cost typename
COST_TYPENAME = 'cost'


class dw_LossLayer(base.LossLayer):
    def set_label(self,label):
        # only care about certain labels
        self.label = label
    def forward(self, bottom, top):
        """Forward emits the loss, and computes the gradient as well."""
        diff = bottom[0].init_diff(setzero=False)
        diff[:] = bottom[0].data()
        print diff
        diff -= bottom[1].data()
        print diff
        diff[set(range(len(diff))).difference(set(self.label))] = 0
        print diff 
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
        print padded_diff.dtype
        print (t_sz[0],t_sz[1],t_sz[2],k_sz[0])
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
    
def dw_data():
    """We will create a dummy imagenet data of one single image."""
    data = np.random.rand(1, 220, 220, 3).astype(np.float32)
    label = np.zeros((1,1000)).astype(np.float32)
    label[0][np.random.randint(1000, size=1)] = 1
    #label = np.random.randint(1000, size=1)
    dataset = core_layers.NdarrayDataLayer(name='input', sources=[data, label])
    return dataset


def dw_cudacnn(cuda_layers, output_shapes):
    decaf_net = dw_Net()
    #decaf_net = base.Net()
    exclude_list = []
    db = 1
    if db:
        #1. data layer
        decaf_net.add_layers(dw_data(),provides=['data', 'label'])
        #2. jeffnet layer
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
            decaf_net.add_layers(decaf_layer, needs=needs, provides=provide)
    if db:
        #3. loss layer
        #loss_layer = core_layers.SquaredLossLayer(name='loss')        
        loss_layer = dw_LossLayer(name='loss')        
        decaf_net.add_layer(loss_layer,needs=['fc8_cudanet_out', 'label'])
    
    decaf_net.finish()
    return decaf_net


cuda_decafnet = pickle.load(open(FLAGS.net_file))
net = dw_cudacnn(cuda_decafnet, {'data': (INPUT_DIM, INPUT_DIM, 3)})

net.layers['loss'].set_label(147)
net.layers['conv1']._large_mem = True

im = net.forward_backward()
import scipy.io
scipy.io.savemat('ha.mat',mdict={'im':im})
def check_update():
    b = net.layers['fc8']._bias.data()
    bd = net.layers['fc8']._bias.diff.im_self.data()
    net.update()
    bb = net.layers['fc8']._bias.data()

"""
meta = pickle.load(open(FLAGS.meta_file))
label_names = meta['label_names']
_data_mean = translator.img_cudaconv_to_decaf(meta['data_mean'], 256, 3)
[id for id,x in enumerate(label_names) if 'pug' in x]
"""

