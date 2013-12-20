import cPickle 
import numpy as np
from pylearn2.models.mlp import MLP
from pylearn2.training_algorithms.sgd import SGD
from DBL_layer import DBL_ConvLayers,DBL_FcLayers,DBL_CfLayers
from DBL_util import *
from DBL_data import DBL_Data

class DBL_model(object):
    def __init__(self,ishape,p_layers,p_data): 
        # 1. data                 
        self.loadParam_layer(p_layers,ishape)                
        self.DataLoader = DBL_Data(p_data)                    

    # 1. I/O
    def loadData(self,basepath,which_set,data_ind=None,options=None):
        self.DataLoader.loadData(basepath,which_set,data_ind,options)
            
    def loadWeight(self, pklname):
         # create DBL_model          
        # load and rebuild model
        layer_params = cPickle.load(open(pklname))
        layer_id = 0
        num_layers = len(self.model.layers)
        for layer in self.model.layers:
            if layer_params[layer_id][0].ndim==2:
                layer.set_weights(layer_params[layer_id][0])
                layer.set_biases(layer_params[layer_id][1])
            else:
                layer.set_weights(layer_params[layer_id][1])
                layer.set_biases(layer_params[layer_id][0])
            
            layer_id = layer_id + 1                            

    def saveWeight(self,pklname):                
        # save the model
        layer_params = []
        for layer in self.model.layers:
            param = layer.get_params()      
            #print param
            #print param[0].get_value().shape
            #print param[1].get_value().shape
            layer_params.append([param[0].get_value(), param[1].get_value()])
            
        cPickle.dump(layer_params, open(pklname, 'wb'))
    
    def loadParam_layer(self,p_layers,ishape):    
        # setup layer
        layers = []
        for param in p_layers:            
            if param[0].param_type==0:
                layers = layers + DBL_ConvLayers(param)
            elif param[0].param_type==1:
                layers = layers + DBL_FcLayers(param)
            elif param[0].param_type==2:
                layers = layers + DBL_CfLayers(param)        
        self.model = MLP(layers, input_space=ishape)

    def loadParam_algo(self,p_algo):
        # setup algo
        self.algo =  SGD(learning_rate = p_algo.learning_rate,
        cost = p_algo.cost,
        batch_size = p_algo.batch_size,
        monitoring_batches = p_algo.monitoring_batches,
        monitoring_dataset = {'valid':self.DataLoader.data['valid']},
        #monitoring_dataset = self.DataLoader.data,
        monitor_iteration_mode = p_algo.monitor_iteration_mode,
        termination_criterion = p_algo.termination_criterion,
        update_callbacks = p_algo.update_callbacks,
        learning_rule = p_algo.learning_rule,
        init_momentum = p_algo.init_momentum,
        set_batch_size = p_algo.set_batch_size,
        train_iteration_mode = p_algo.train_iteration_mode,
        batches_per_iter = p_algo.batches_per_iter,
        theano_function_mode = p_algo.theano_function_mode,
        monitoring_costs = p_algo.monitoring_costs,
        seed = p_algo.seed)
        self.algo.setup(self.model, self.DataLoader.data['train'])


    def train(self,p_algo):        
        self.loadParam_algo(p_algo)
        while True:
            #print d_train.X.shape,d_train.y.shape
            self.algo.train(self.DataLoader.data['train'])
            self.model.monitor.report_epoch()            
            self.model.monitor()
            """
            # hack the monitor
            print "monior:\n"
            self.test(self.ds_valid)
            """
            if not self.algo.continue_learning(self.model):
                break    

    def test(self,batch_size,metric=0):
        """
        metric: evaluation metric
        0: classfication error
        1: L1 regression error
        2: L2 regression error
        """
        data_test = self.DataLoader.data['test']
        
        # make batches
        batch_size = batch_size
        m = data_test.X.shape[0]
        extra = (batch_size - m % batch_size) % batch_size
        #print extra,batch_size,m
        assert (m + extra) % batch_size == 0
        if extra > 0:
            data_test.X = np.concatenate((data_test.X, np.zeros((extra, data_test.X.shape[1]),
                    dtype=data_test.X.dtype)), axis=0)
            assert data_test.X.shape[0] % batch_size == 0
        X = self.model.get_input_space().make_batch_theano()
        Y = self.model.fprop(X)
        
        from theano import function
        if metric==0:
            from theano import tensor as T
            y = T.argmax(Y, axis=1)        
            f = function([X], y)
        else:
            f = function([X], Y)
        
        yhat = []
        for i in xrange(data_test.X.shape[0] / batch_size):
            x_arg = data_test.X[i*batch_size:(i+1)*batch_size,:]
            if X.ndim > 2:
                x_arg = data_test.get_topological_view(x_arg)
            yhat.append(f(x_arg.astype(X.dtype)))

        yhat = np.concatenate(yhat)
        yhat = yhat[:m]
        data_test.X = data_test.X[:m,:]
        y = data_test.y
        if metric == 0:
            if data_test.y.ndim>1:
                y = np.argmax(data_test.y,axis=1)
            assert len(y)==len(yhat)
            acc = float(np.sum(y-yhat==0))/y.size
        elif metric == 1:
            acc = float(np.sum(abs(y-yhat)))/y.size
        elif metric == 2: 
            acc = float(np.sum((y-yhat)**2))/y.size
        
        return [[y],[acc]]


