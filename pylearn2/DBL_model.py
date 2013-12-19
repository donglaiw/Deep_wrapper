from DBL_util import *
import numpy as np
from pylearn2.models.mlp import MLP
from pylearn2.training_algorithms.sgd import SGD
from DBL_layer import DBL_ConvLayers,DBL_FcLayers,DBL_CfLayers

class DBL_model(object):
    def __init__(self,ishape,p_layers,p_algo,data): 
        # 1. data 
        self.data = data
        self.setup_layer(p_layers,ishape)        
        self.setup_algo(p_algo)

    

    def setup_layer(self,p_layers,ishape):    
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

    def setup_algo(self,p_algo):
        # setup algo
        self.algo =  SGD(learning_rate = p_algo.learning_rate,
        cost = p_algo.cost,
        batch_size = p_algo.batch_size,
        monitoring_batches = p_algo.monitoring_batches,
        monitoring_dataset = self.data['valid'],
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
        self.algo.setup(self.model, self.data['train'])


    def train(self):
        while True:
            #print d_train.X.shape,d_train.y.shape
            self.algo.train(self.data['train'])
            self.model.monitor.report_epoch()            
            self.model.monitor()
            """
            # hack the monitor
            print "monior:\n"
            self.test(self.ds_valid)
            """
            if not self.algo.continue_learning(self.model):
                break    
    
    def test(self,ds2):
        # https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/icml_2013_wrepl/emotions/make_submission.py
        batch_size = 500 #self.algo.batch_size
        m = ds2.X.shape[0]
        extra = (batch_size - m % batch_size) % batch_size
        #print extra,batch_size,m
        assert (m + extra) % batch_size == 0
        if extra > 0:
            ds2.X = np.concatenate((ds2.X, np.zeros((extra, ds2.X.shape[1]),
                    dtype=ds2.X.dtype)), axis=0)
            assert ds2.X.shape[0] % batch_size == 0
        X = self.model.get_input_space().make_batch_theano()
        Y = self.model.fprop(X)

        from theano import tensor as T
        y = T.argmax(Y, axis=1)
        from theano import function
        f = function([X], y)
        y = []
        for i in xrange(ds2.X.shape[0] / batch_size):
            x_arg = ds2.X[i*batch_size:(i+1)*batch_size,:]
            if X.ndim > 2:
                x_arg = ds2.get_topological_view(x_arg)
            y.append(f(x_arg.astype(X.dtype)))

        y = np.concatenate(y)
        y = y[:m]
        ds2.X = ds2.X[:m,:]
        """
        print y
        print ds2.y
        
        """
        if ds2.y.ndim>1:
            yy = np.argmax(ds2.y,axis=1)
        else:
            yy = ds2.y
        print len(y)
        print len(yy)
        acc = 0
        if len(yy)>0: 
            assert len(y)==len(yy)
            acc = float(np.sum(y-yy==0))/len(yy)
        print acc
        return [[y],[acc]]


