import sys
VLIB='/home/Stephen/Desktop/VisionLib/'
sys.path.append(VLIB+'Donglai/DeepL/pylearn2')
from pylearn2.space import Conv2DSpace,VectorSpace
from pylearn2.models.mlp import Softmax,MLP
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter
from DBL_layer import *
from DBL_model import DBL_model
from DBL_data import *

import os
import cPickle
import numpy as np


class CNN_NET():
    def __init__(self, basepath,ishape, p_model, p_algo):        
        self.basepath = basepath
        self.ishape = ishape 
        self.nclass = p_model[-1][-1].n_classes
        self.p_model = p_model
        self.p_algo = p_algo

    def loaddata(self,id_data,ind_train,ind_valid,id_pre=0):
        """
        from pylearn2.datasets.preprocessing import GlobalContrastNormalization
        pre = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)            
        """
        pre = None
        if id_pre == 1:
            from pylearn2.datasets.preprocessing import GlobalContrastNormalization
            pre = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)    
    	if id_data==0:
    	    self.data_train = Occ('train',self.nclass,self.basepath,ind_train,pre,self.ishape)    
    	    self.data_valid = Occ('train',self.nclass,self.basepath,ind_valid,pre,self.ishape)
    	elif id_data==1:
    	    self.data_train = ICML_emotion('train',self.nclass,self.basepath,ind_train,pre,self.ishape)    
    	    self.data_valid = ICML_emotion('train',self.nclass,self.basepath,ind_valid,pre,self.ishape)    
        elif id_data==2:
            self.data_train = Denoise('train',self.nclass,self.basepath,ind_train,pre,self.ishape)    
            self.data_valid = Denoise('train',self.nclass,self.basepath,ind_valid,pre,self.ishape)    

    def model(self,id_pre=0):
        # create conv layers
        layers = DBL_layers(self.p_model)      
        # print layers.layers
        model = MLP(layers.layers, input_space=self.ishape)
        if id_pre == 1:
            model.layers[-1].init_bias_target_marginals=self.data_train            

        algo_term = EpochCounter(self.p_algo.num_epoch) # number of epoch iteration
        algo = SGD(learning_rate = self.p_algo.rate_grad,
                batch_size = self.p_algo.num_perbatch,
                init_momentum = self.p_algo.rate_momentum,
                monitoring_dataset = self.data_valid,
                termination_criterion=algo_term
                )   

        self.DBL = DBL_model(model,algo,self.data_train)

    def train(self,pklname='tmp.pkl'):
        # load data
        #print data_valid.X.shape,data_valid.y.shape
        #print data_train.X.shape,data_train.y.shape

        self.DBL.train()

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
	# e.g. denoising 
    #ishape = VectorSpace(17,17,3)
    nclass = 17*17

    DD = '/home/Stephen/Desktop/Deep_Low/dn/'
    param = param_dnn()
    
    ishape = Conv2DSpace(shape = (17,17),num_channels = 3)        
    """"""
    p_fc = param.param_model_fc(dim = 1000,irange=0.1)    
    p_cf = param.param_model_cf(n_classes = nclass,irange=0.1)        
    p_algo = param.param_algo(num_perbatch = 1000,num_epoch=100,rate_grad=0.001,rate_momentum=0.5)
                            

    net = CNN_NET(DD, ishape, [[p_fc],[p_cf]],p_algo)
    
    np.random.seed(1)
    rand_ind = np.random.permutation([i for i in range(100000)])
    net.loaddata(2,rand_ind[:90000],rand_ind[90000:])
    net.model()
    net.train()
    #net.test()
