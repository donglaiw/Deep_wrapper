import sys
VLIB='/home/Stephen/Desktop/VisionLib/'
sys.path.append(VLIB+'Donglai/DeepL/deepmodel/pylearn2')

from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import Softmax,MLP
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter

from DBL_layer import DBL_ConvLayers
from DBL_model import DBL_model
from DBL_data import *

import os
import cPickle
import numpy as np


class P_cnn():
    def __init__(self,basepath,im_sz,num_class):
        self.ishape = Conv2DSpace(
                shape = im_sz,
                num_channels = 1
                )    
        self.nclass = num_class

    def loaddata(self,data_set,train,valid,id_data):
        from pylearn2.datasets.preprocessing import GlobalContrastNormalization
        pre = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)    
    	if id_data==0:
    	    self.data_train = Occ('train',nclass,basepath,0,10000,pre,1,ishape)    
    	    self.data_valid = Occ('train',nclass,basepath,10000,15000,pre,2,ishape)
    	elif id_data==1:
    	    data_train = ICML_emotion('train',nclass,basepath,0,10000,pre,1,ishape)    
    	    data_valid = ICML_emotion('train',nclass,basepath,10000,11000,pre,2,ishape)    

    class param_model_conv():
        def __init__(self,nkernels,kshape,irange,pshape,pstride,knorm,layer_type=0):
            self.lt = layer_type
            self.nk = nkernels
            self.ks = kshape
            self.ir = irange
            self.ps = pshape
            self.pd = pstride
            self.kn = knorm
    def model(self,id_model):
        # create conv layers
        layers = [];
        layers = DBL_ConvLayers(self.para_model_conv)   
        if id_model==0:
            nk = [30]
            nk = [20,30]
            ks = [[5,5],[5,5],[3,3]]
            ir = [0.05,0.05,0.05]
            ps = [[4,4],[4,4],[2,2]]
            pd = [[2,2],[2,2],[2,2]]
            kn = [0.9,0.9,0.9]
            
        elif id_model == 1:
            h1 = Softmax(
                layer_name='h1',
                #max_col_norm = 1.9365,
                n_classes = nclass,
                init_bias_target_marginals=None,
                #istdev = .05
                irange = .0
            )
            layers.append(layer_soft)     

        self.model = MLP(layers, input_space=ishape)
        self.DBL = DBL_model(basepath,nclass,np.append(ishape.shape,1),preproc,cutoff)    

    class param_train():
        def __init__(self,num_perbatch,num_epoch=100,rate_grad=0.001,rate_momentum=0.5):
            self.num_perbatch   = num_perbatch
            self.num_epoch      = num_epoch
            self.rate_grad      = rate_grad
            self.rate_momentum  = rate_momentum

    def train(self,id_train=0; cutoff=[-1,-1],preproc=[0,0],pklname='tmp.pkl'):
        # load data
        #print data_valid.X.shape,data_valid.y.shape
        #print data_train.X.shape,data_train.y.shape
        if id_train==0:
            self.model.layers[-1].init_bias_target_marginals=self.data_train            
        elif id_train=1:
            from pylearn2.datasets.preprocessing import GlobalContrastNormalization
            pre = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)    
            param = param_train(500,500,0.001,0.5)


        algo_term = EpochCounter(param.num_epoch) # number of epoch iteration
        algo = SGD(learning_rate = param.rate_grad,
                batch_size = param.num_perbatch,
                init_momentum = param.rate_momentum,
                monitoring_dataset = data_valid,
                termination_criterion=algo_term
                )   
        self.DBL.run_model(self.model,algo,data_train)

        # save the model
        if pklname!='':
            layer_params = []
            for layer in layers:
                param = layer.get_params()      
                print param
                print param[0].get_value()
                layer_params.append([param[0].get_value(), param[1].get_value()])
                
            #cPickle.dump(DBL,open(pklname, 'wb'))
            #cPickle.dump(layer_params, open(pklname + '.cpu', 'wb'))
            cPickle.dump(layer_params, open(pklname + '.cpu', 'wb'))

    def test(pklname):
    	 # create DBL_model          
        # load and rebuild model
        layer_params = cPickle.load(open(pklname + '.cpu'))
        layer_id = 0
        for layer in model.layers:
            if layer_id < len(layers) - 1:
                layer.set_weights(layer_params[layer_id][0])
                layer.set_biases(layer_params[layer_id][1])
            else:
                layer.set_weights(layer_params[layer_id][1])
                layer.set_biases(layer_params[layer_id][0])
            
            layer_id = layer_id + 1
        
        DBL.model = model                        

if __name__ == "__main__": 
	
    DD = '/home/Stephen/Desktop/Edge/Hack'
    net = P_cnn()
    #p_conv = net.param_conv(30,(5,5),0.05,(4,4),(2,2),0.9)
    

    p_train = net.param_train(500,500,0.01,0.5)

	DD = '/home/Stephen/Desktop/Data/Classification/icml_2013_emotions'
	T1_v,T1_t = train2(DD)
	"""
