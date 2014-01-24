import cPickle 
import numpy as np
from pylearn2.models.mlp import MLP
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.bgd import BGD
from DBL_layer import DBL_ConvLayers,DBL_FcLayers,DBL_CfLayers
from DBL_util import trainMonitor,paramSet
from DBL_data import DBL_Data
import scipy.io
import glob,os


class DBL_model(object):
    def __init__(self,algo_id,model_id,num_epoch,num_dim,test_id): 
        self.algo_id = algo_id
        self.model_id = model_id
        self.num_epoch = num_epoch
        self.num_dim = num_dim
        self.test_id = test_id

        self.path_train = None
        self.path_test = None
        self.p_data = None
        self.batch_size = None
        self.do_savew = True

        self.param = paramSet()
        self.p_monitor = {}
    def loadData(self,basepath,which_set,data_ind=None):
        self.DataLoader.loadData(self.p_data,basepath,which_set,data_ind)
            
    def loadWeight(self, fname):
         # create DBL_model          
        # load and rebuild model
        if fname[-3:] == 'pkl':
            layer_params = cPickle.load(open(fname))
        elif fname[-3:] == 'mat':
            mat = scipy.io.loadmat(fname)            
            layer_params = mat['param']            
        else:
            raise('cannot recognize: '+fname)

        layer_id = 0
        num_layers = len(self.model.layers)
        for layer in self.model.layers:
            # squeeze for matlab structure
            if np.squeeze(layer_params[layer_id][0]).ndim==2:
                layer.set_weights(layer_params[layer_id][0])
                layer.set_biases(np.squeeze(layer_params[layer_id][1]))
                #tmp = np.squeeze(layer_params[layer_id][1])                
            else:
                layer.set_weights(layer_params[layer_id][1])
                layer.set_biases(np.squeeze(layer_params[layer_id][0]))
                #tmp = np.squeeze(layer_params[layer_id][0])
            #print "aa:",layer_params[layer_id][1].shape,layer_params[layer_id][0].shape
            #print "sss:",layer_params[layer_id][1][:10]
            #print "ttt:",layer_params[layer_id][0][0]
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
    
    def loadAlgo(self,p_algo):
        # setup algo
        print self.DataLoader.data
        if p_algo.algo_type==0:
            self.algo =  SGD(learning_rate = p_algo.learning_rate,
            cost = p_algo.cost,
            batch_size = p_algo.batch_size,
            monitoring_batches = p_algo.monitoring_batches,
            monitoring_dataset = p_algo.monitoring_dataset,
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
        elif p_algo.algo_type==1:
                self.algo = BGD(
                cost = p_algo.cost,
                batch_size=p_algo.batch_size, 
                batches_per_iter=p_algo.batches_per_iter,
                updates_per_batch = p_algo.updates_per_batch,
                monitoring_batches=p_algo.monitoring_batches,
                monitoring_dataset=p_algo.monitoring_dataset,
                termination_criterion =p_algo.termination_criterion, 
                set_batch_size = p_algo.set_batch_size,
                reset_alpha = p_algo.reset_alpha, 
                conjugate = p_algo.conjugate,
                min_init_alpha = p_algo.min_init_alpha,
                reset_conjugate = p_algo.reset_conjugate, 
                line_search_mode = p_algo.line_search_mode,
                verbose_optimization=p_algo.verbose_optimization, 
                scale_step=p_algo.scale_step, 
                theano_function_mode=p_algo.theano_function_mode,
                init_alpha = p_algo.init_alpha, 
                seed = p_algo.seed)
        self.algo.setup(self.model, self.DataLoader.data['train'])

    def setup(self):
        self.setupParam()
        self.check_setupParam()

        self.dl_id = str(self.algo_id)+'_'+str(self.model_id)+'_'+str(self.num_dim).strip('[]').replace(', ','_')+'_'+str(self.test_id)+'_'+str(self.num_epoch)
        self.param_pkl = 'dl_p'+self.dl_id+'.pkl'
        self.result_mat = 'result/'+self.dl_id+'/dl_r'+str(self.test_id)+'.mat'
        self.buildModel()
        self.buildLayer()                
        
        self.DataLoader = DBL_Data()
        self.do_test = True
        if not os.path.exists(self.param_pkl):
            self.do_test = False
            # training
            self.loadData_train()
            self.buildAlgo()


    def setupParam(self):
        raise NotImplementedError(str(type(self)) + " does not implement: setupParam().")
    def check_setupParam(self):
        varnames = ['path_train','path_test','p_data','batch_size']
        for varname in varnames:
            if eval('self.'+varname+'== None'):
                raise ValueError('Need to set "'+varname+'" in setupParam()')
    def buildModel(self):
        raise NotImplementedError(str(type(self)) + " does not implement: buildModel().")
    def buildAlgo(self):
        raise NotImplementedError(str(type(self)) + " does not implement: buildAlgo().")
    def train(self):
        raise NotImplementedError(str(type(self)) + " does not implement: train().")
    def test(self):
        raise NotImplementedError(str(type(self)) + " does not implement: test().")
    def loadData_train(self):
        raise NotImplementedError(str(type(self)) + " does not implement: buildAlgo().")
    def run(self):
        if self.do_test:
            self.test()
        else:
            # training
            self.train()
    
    def buildLayer(self):    
        # setup layer
        self.layers = []
        for param in self.p_layers:            
            if param[0].param_type==0:
                self.layers = self.layers + DBL_ConvLayers(param)
            elif param[0].param_type==1:
                self.layers = self.layers + DBL_FcLayers(param)
            elif param[0].param_type==2:
                self.layers = self.layers + DBL_CfLayers(param)        
        self.model = MLP(self.layers, input_space=self.ishape)

        # load available weight
        pre_dl_id = self.param_pkl[:self.param_pkl.rfind('_')+1]
        fns = glob.glob(pre_dl_id+'*.pkl')
        epoch_max = 0
        if len(fns)==0:
            # first time to do it, load matlab prior
            mat_init = 'init_p'+str(self.model_id)+'.mat'
            if os.path.exists(mat_init):
                print "load initial mat weight: ", mat_init
                self.loadWeight(mat_init)
        else:
            for fn in fns:
                epoch_id = int(fn[fn.rfind('_')+1:fn.find('.pkl')])
                if (epoch_id>epoch_max and epoch_id<=self.num_epoch):
                    epoch_max = epoch_id
            if epoch_max>0:
                print "load weight at epoch: ", epoch_max
                self.loadWeight(pre_dl_id+str(epoch_max)+'.pkl')
                self.num_epoch -= epoch_max
        self.p_monitor['epoch'] = epoch_max

    def runTrain(self):        
        self.loadAlgo(self.p_algo)
        self.train_monitor = trainMonitor(self.model.monitor,self.p_monitor)
        #self.model.monitor.report_epoch()            
        self.train_monitor.run()
        while self.algo.continue_learning(self.model):
            self.algo.train(self.DataLoader.data['train'])            
            self.train_monitor.run()
            if self.do_savew and (self.train_monitor.monitor._epochs_seen+1)%10 == 0:
                self.saveWeight(self.param_pkl)
            #self.model.monitor()            
        if self.do_savew:
            self.saveWeight(self.param_pkl)


    def runTest(self,data_test=None,metric=0):
        """
        metric: evaluation metric
        0: classfication error
        1: L1 regression error
        2: L2 regression error
        """
        if data_test == None:
            data_test = self.DataLoader.data['test']
        batch_size = self.batch_size
        # make batches
        m = data_test.X.shape[0]
        extra = (batch_size - m % batch_size) % batch_size
        #print extra,batch_size,m
        assert (m + extra) % batch_size == 0
        #print data_test.X[0]
        if extra > 0:
            data_test.X = np.concatenate((data_test.X, np.zeros((extra, data_test.X.shape[1]),
                    dtype=data_test.X.dtype)), axis=0)
            assert data_test.X.shape[0] % batch_size == 0
        X = self.model.get_input_space().make_batch_theano()
        Y = self.model.fprop(X)
        """
        print 'load param:'
        param = self.model.layers[0].get_params()
        aa = param[0].get_value()
        bb = param[1].get_value()
        print aa[:3,:3],bb[:10]   
        """
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
        #print "ww:",x_arg.shape
        yhat = np.concatenate(yhat)
        yhat = yhat[:m]
        data_test.X = data_test.X[:m,:]
        y = data_test.y
        #print m,extra
        acc = -1
        if y != None:
            if metric == 0:
                if data_test.y.ndim>1:
                    y = np.argmax(data_test.y,axis=1)
                assert len(y)==len(yhat)
                acc = float(np.sum(y-yhat==0))/m
            elif metric == 1:
                acc = float(np.sum(abs(y-yhat)))/m
            elif metric == 2: 
                #print y.shape,yhat.shape,float(np.sum((y-yhat)**2)),y.size
                #print y[:,0]
                #print "yhat: ",yhat[0]
                #print float(np.sum((y-yhat)**2))
                acc = float(np.sum((y-yhat)**2))/m
            
        return [[yhat],[acc]]


