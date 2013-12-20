import sys
VLIB='/home/Stephen/Desktop/VisionLib/'
sys.path.append(VLIB+'Donglai/DeepL/pylearn2')
from pylearn2.space import Conv2DSpace,VectorSpace

from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.termination_criteria import EpochCounter

from DBL_model import DBL_model
from DBL_data import *
from DBL_util import *
import os
import cPickle
import numpy as np


class CNN_NET():
    def __init__(self,ishape):        
        self.ishape = ishape

    def loaddata(self,basepath,id_data,nclass,ind_train,ind_valid,id_pre=0):
        """
        from pylearn2.datasets.preprocessing import GlobalContrastNormalization
        pre = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)            
        """
        print "load"
        pre = None
        if id_pre == 1:
            from pylearn2.datasets.preprocessing import GlobalContrastNormalization
            pre = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)    

    	if id_data==0:
    	    self.data_train = Occ('train',nclass,basepath,ind_train,pre,self.ishape)    
    	    self.data_valid = Occ('train',nclass,basepath,ind_valid,pre,self.ishape)
    	elif id_data==1:
    	    self.data_train = ICML_emotion('train',nclass,basepath,ind_train,pre,self.ishape)    
    	    self.data_valid = ICML_emotion('train',nclass,basepath,ind_valid,pre,self.ishape)    
        elif id_data==2:
            print self.ishape
            self.data_train = Denoise('train',nclass,basepath,ind_train,pre,self.ishape)    
            self.data_valid = Denoise('train',nclass,basepath,ind_valid,pre,self.ishape)    

    def setup(self, p_layers, p_algo):
        # create conv layers        
        self.DBL = DBL_model(self.ishape, 
                        p_layers,
                        p_algo,{'valid': self.data_valid,
                                'train': self.data_train})

    def train(self,pklname='tmp.pkl'):
        # load data
        #print data_valid.X.shape,data_valid.y.shape
        #print data_train.X.shape,data_train.y.shape

        self.DBL.train()

        # save the model
        if pklname!='':
            layer_params = []
            for layer in self.DBL.layers:
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
        

if __name__ == "__main__": 
	# e.g. denoising 
    #ishape = VectorSpace(17,17,3)
    nclass = 17*17    
    DD = './voc/'
    param = SetParam()
    
    ishape = Conv2DSpace(shape = (17,17),num_channels = 3)        
    """"""
    p_fc = param.param_model_fc(dim = 1000,irange=0.1)    
    p_cf = param.param_model_cf(n_classes = nclass,irange=0.1)        
    p_algo = param.param_algo(batch_size = 1000,
                             termination_criterion=EpochCounter(max_epochs=475),
                              cost=Dropout(input_include_probs={'l1': .8},
                                     input_scales={'l1': 1.}),                             
                             learning_rate=0.001,
                             init_momentum=0.5)
                            
    net = CNN_NET(DD)
    
    np.random.seed(1)
    rand_ind = np.random.permutation([i for i in range(100000)])
    net.loaddata(2,rand_ind[:90000],rand_ind[90000:])
    
    net.setup(ishape, [[p_fc],[p_cf]],p_algo)
    
    net.train()
