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

    def loaddata(self,id_data):
        """
        from pylearn2.datasets.preprocessing import GlobalContrastNormalization
        pre = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)            
        """
        pre = None
    	if id_data==0:
    	    self.data_train = Occ('train',nclass,self.basepath,0,10000,pre,1,ishape)    
    	    self.data_valid = Occ('train',nclass,self.basepath,10000,15000,pre,2,ishape)
    	elif id_data==1:
    	    data_train = ICML_emotion('train',nclass,self.basepath,0,10000,pre,1,self.ishape)    
    	    data_valid = ICML_emotion('train',nclass,self.basepath,10000,11000,pre,2,self.ishape)    
        elif id_data==2:
            np.random.seed(1)
            rand_ind = np.random.permutation([i for i in range(100000)])
            data_train = Denoise('train',nclass,self.basepath,rand_ind[:90000],pre,1,self.ishape)    
            data_valid = Denoise('train',nclass,self.basepath,rand_ind[10000:],pre,2,self.ishape)    

    def model(self):
        # create conv layers
        layers = DBL_layers(self.p_model)      
        # print layers.layers
        self.model = MLP(layers.layers, input_space=self.ishape)
        self.DBL = DBL_model(self.basepath,nclass,np.append(ishape.shape,1),preproc,cutoff)    


    def train(self, id_pre, cutoff=[-1,-1],preproc=[0,0],pklname='tmp.pkl'):
        # load data
        #print data_valid.X.shape,data_valid.y.shape
        #print data_train.X.shape,data_train.y.shape
        if id_pre == 0:
            self.model.layers[-1].init_bias_target_marginals=self.data_train            
        elif id_pre == 1:
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
    
    net.loaddata(2)
    net.model()
    net.train()
    #net.test()
