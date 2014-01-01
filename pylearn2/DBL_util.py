
import time
import numpy as np

"""
import logging
log = logging.getLogger(__name__)
"""

from pylearn2.utils import safe_izip
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils import function
from theano import config
from theano.compat.python2x import OrderedDict
from theano import tensor as T
class paramSet():
    # wrapper to create stacked layers
    def __init__(self):
        pass        
    class param_algo():
        def __init__(self,
            learning_rate=1e-3, 
            cost=None, 
            batch_size=None,
            monitoring_batches=None, 
            monitoring_dataset=None,
            monitor_iteration_mode='sequential',
            termination_criterion=None, 
            update_callbacks=None,
            learning_rule = None, 
            init_momentum = None, 
            set_batch_size = False,
            train_iteration_mode = None, 
            batches_per_iter=None,
            theano_function_mode = None, 
            monitoring_costs=None,
            seed=[2012, 10, 5],
            # bgd:
            updates_per_batch = 10,
            reset_alpha=True,
            conjugate = False,
            min_init_alpha=.001, 
            reset_conjugate=True,
            line_search_mode=None, 
            verbose_optimization=False,
            scale_step=1., 
            init_alpha=None,            
            algo_type = 0):
            
            # shared
            self.cost = cost
            self.batch_size = batch_size
            self.monitoring_batches = monitoring_batches
            self.monitoring_dataset = monitoring_dataset
            self.monitor_iteration_mode = monitor_iteration_mode
            self.termination_criterion = termination_criterion
            self.update_callbacks = update_callbacks
            self.init_momentum = init_momentum
            self.set_batch_size = set_batch_size
            self.train_iteration_mode = train_iteration_mode
            self.batches_per_iter = batches_per_iter
            self.theano_function_mode = theano_function_mode
            self.monitoring_costs = monitoring_costs
            self.seed = seed
            self.algo_type = algo_type
            if algo_type == 0:
                # sgd:
                self.learning_rate = learning_rate
                self.learning_rule = learning_rule
            else:
                # bgd:
                self.updates_per_batch = updates_per_batch
                self.reset_alpha=reset_alpha
                self.conjugate = conjugate
                self.min_init_alpha=min_init_alpha
                self.reset_conjugate=reset_conjugate
                self.line_search_mode=line_search_mode
                self.verbose_optimization=verbose_optimization
                self.scale_step=scale_step
                self.init_alpha=init_alpha

    class param_model_conv():
        def __init__(self,
                 output_channels,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 irange = None,
                 border_mode = 'valid',
                 sparse_init = None,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 left_slope = 0.0,
                 max_kernel_norm = None,
                 pool_type = 'max',
                 detector_normalization = None,
                 output_normalization = None,
                 kernel_stride=(1, 1),
                 layer_type=0):
            self.output_channels=output_channels
            self.kernel_shape=kernel_shape
            self.pool_shape=pool_shape
            self.pool_stride=pool_stride
            self.irange = irange,
            self.border_mode = border_mode
            self.sparse_init = sparse_init
            self.include_prob = include_prob
            self.init_bias = init_bias
            self.W_lr_scale = W_lr_scale
            self.b_lr_scale = b_lr_scale
            self.left_slope = left_slope
            self.max_kernel_norm = max_kernel_norm
            self.pool_type = pool_type
            self.detector_normalization = detector_normalization
            self.output_normalization = output_normalization
            self.kernel_stride = kernel_stride
            self.layer_type = layer_type
            self.param_type = 0
    class param_model_fc():
        def __init__(self, dim,
                 irange = None,
                 istdev = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 mask_weights = None,
                 max_row_norm = None,
                 max_col_norm = None,
                 softmax_columns = False,
                 copy_input = 0,
                 use_abs_loss = False,
                 use_bias = True,
                 layer_type=0):
            self.dim = dim
            self.irange = irange
            self.istdev = istdev
            self.sparse_init = sparse_init
            self.sparse_stdev = sparse_stdev
            self.include_prob = include_prob
            self.init_bias = init_bias
            self.W_lr_scale = W_lr_scale
            self.b_lr_scale = b_lr_scale
            self.mask_weights = mask_weights
            self.max_row_norm = max_row_norm
            self.max_col_norm = max_col_norm
            self.softmax_columns = softmax_columns
            self.copy_input = copy_input
            self.use_abs_loss = use_abs_loss
            self.use_bias = use_bias
            self.layer_type = layer_type
            self.param_type = 1
    class param_model_cf():
        def __init__(self, n_classes, 
                irange = None,
                istdev = None,
                sparse_init = None, 
                W_lr_scale = None,
                b_lr_scale = None, 
                max_row_norm = None,
                no_affine = False,
                max_col_norm = None, 
                init_bias_target_marginals= None,
                layer_type=0):
                self.n_classes = n_classes
                self.irange = irange
                self.istdev = istdev
                self.sparse_init = sparse_init 
                self.W_lr_scale = W_lr_scale 
                self.b_lr_scale = b_lr_scale  
                self.max_row_norm = max_row_norm 
                self.no_affine = no_affine 
                self.max_col_norm = max_col_norm  
                self.init_bias_target_marginals= init_bias_target_marginals
                self.layer_type = layer_type
                self.param_type = 2

class trainMonitor():
    def __init__(self,monitor,p_monitor):
        self.monitor = monitor        
        self.monitor._build_data_specs()        
        init_names = dir(self)
        self.monitor.prereqs = OrderedDict()
        for channel in self.monitor.channels.values():
            if channel.prereqs is not None:
                dataset = channel.dataset
                if dataset not in self.monitor.prereqs:
                    self.monitor.prereqs[dataset] = []
                prereqs = self.monitor.prereqs[dataset]
                for prereq in channel.prereqs:
                    if prereq not in prereqs:
                        prereqs.append(prereq)
                
        self.p_channel = p_monitor['channel']        
        self.p_save = p_monitor['save'] if 'save' in p_monitor else None
        self.monitor._epochs_seen = p_monitor['epoch']
         
        """
        # screw up theano args
        for channel in self.monitor.channels.keys():
            if channel not in p_channel:
                del self.monitor.channels[channel]
        """
    
    def run(self):
        mm = self.monitor        

        updates = OrderedDict()
        for channel in mm.channels.values():
            updates[channel.val_shared] = np.cast[config.floatX](0.0)        
        mm.begin_record_entry = function(inputs=[], updates=updates, mode=mm.theano_function_mode,
                    name = 'Monitor.begin_record_entry')


        updates = OrderedDict()
        givens = OrderedDict()
        theano_args = mm._flat_data_specs[0].make_theano_batch(
                ['monitoring_%s' % s for s in mm._flat_data_specs[1]])

        # Get a symbolic expression of the batch size
        # We do it here, rather than for each channel, because channels with an
        # empty data_specs do not use data, and are unable to extract the batch
        # size. The case where the whole data specs is empty is not supported.
        batch_size = mm._flat_data_specs[0].batch_size(theano_args)
        
        nested_theano_args = mm._data_specs_mapping.nest(theano_args)
        if not isinstance(nested_theano_args, tuple):
            nested_theano_args = (nested_theano_args,)        
        
        assert len(nested_theano_args) == (len(mm.channels) + 1)

        for key in sorted(mm.channels.keys()):
            mode = mm.theano_function_mode
            if mode is not None and hasattr(mode, 'record'):
                mode.record.handle_line('compiling monitor including channel '+key+'\n')
            #log.info('\t%s' % key)
        it = [d.iterator(mode=i, num_batches=n, batch_size=b,
                         data_specs=mm._flat_data_specs,
                         return_tuple=True) \
              for d, i, n, b in safe_izip(mm._datasets, mm._iteration_mode,
                                    mm._num_batches, mm._batch_size)]
        mm.num_examples = [np.cast[config.floatX](float(i.num_examples)) for i in it]


        givens = [OrderedDict() for d in mm._datasets]
        updates = [OrderedDict() for d in mm._datasets]

        #for i, channel in enumerate(mm.channels.values()):
        for i, dw_name in enumerate(mm.channels.keys()):            
            if dw_name in self.p_channel:
                channel = mm.channels[dw_name]
                
                index = mm._datasets.index(channel.dataset)
                d = mm._datasets[index]
                g = givens[index]
                cur_num_examples = mm.num_examples[index]
                u = updates[index]

                # Flatten channel.graph_input and the appropriate part of
                # nested_theano_args, to iterate jointly over them.
                c_mapping = DataSpecsMapping(channel.data_specs)
                channel_inputs = c_mapping.flatten(channel.graph_input,
                                                   return_tuple=True)                
                inputs = c_mapping.flatten(nested_theano_args[i + 1],
                                           return_tuple=True)

                for (channel_X, X) in safe_izip(channel_inputs, inputs):
                    assert channel_X not in g or g[channel_X] is X
                    #print channel_X.type , X.type
                    assert channel_X.type == X.type
                    g[channel_X] = X

                if batch_size == 0:
                    # No channel does need any data, so there is not need to
                    # average results, and we will call the accum functions only
                    # once.
                    # TODO: better handling of channels not needing data when
                    # some other channels need data.
                    assert len(mm._flat_data_specs[1]) == 0
                    val = channel.val
                else:
                    if n == 0:
                        raise ValueError("Iterating over 0 examples results in divide by 0")
                    val = (channel.val * T.cast(batch_size, config.floatX)
                            / cur_num_examples)
                u[channel.val_shared] = channel.val_shared + val
            
        mm.accum = []
        for idx, packed in enumerate(safe_izip(givens, updates)):
            g, u = packed
            mode = mm.theano_function_mode
            if mode is not None and hasattr(mode, 'record'):
                for elem in g:
                    mode.record.handle_line('g key '+var_descriptor(elem)+'\n')
                    mode.record.handle_line('g val '+var_descriptor(g[elem])+'\n')
                for elem in u:
                    mode.record.handle_line('u key '+var_descriptor(elem)+'\n')
                    mode.record.handle_line('u val '+var_descriptor(u[elem])+'\n')
            function_name = 'Monitor.accum[%d]' % idx
            if mode is not None and hasattr(mode, 'record'):
                mode.record.handle_line('compiling supervised accum\n')
            # Some channels may not depend on the data, ie, they might just monitor the model
            # parameters, or some shared variable updated by the training algorithm, so we
            # need to ignore the unused input error
            mm.accum.append(function(theano_args,
                                       givens=g,
                                       updates=u,
                                       mode=mm.theano_function_mode,
                                       name=function_name))
        for a in mm.accum:
            if mode is not None and hasattr(mode, 'record'):
                for elem in a.maker.fgraph.outputs:
                    mode.record.handle_line('accum output '+var_descriptor(elem)+'\n')
            #log.info("graph size: %d" % len(a.maker.fgraph.toposort()))
            datasets = mm._datasets
            
        # Set all channels' val_shared to 0
        mm.begin_record_entry()
        
        for d, i, b, n, a, sd, ne in safe_izip(datasets,
                                               mm._iteration_mode,
                                               mm._batch_size,
                                               mm._num_batches,
                                               mm.accum,
                                               mm._rng_seed,
                                               mm.num_examples):
            myiterator = d.iterator(mode=i,
                                    batch_size=b,
                                    num_batches=n,
                                    data_specs=mm._flat_data_specs,
                                    return_tuple=True,
                                    rng=sd)

            # If mm._flat_data_specs is empty, no channel needs data,
            # so we do not need to call the iterator in order to average
            # the monitored values across different batches, we only
            # have to call them once.
            if len(mm._flat_data_specs[1]) == 0:
                X = ()
                mm.run_prereqs(X, d)
                a(*X)

            else:
                actual_ne = 0
                for X in myiterator:
                    # X is a flat (not nested) tuple
                    mm.run_prereqs(X, d)
                    a(*X)
                    actual_ne += mm._flat_data_specs[0].np_batch_size(X)
                # end for X
                if actual_ne != ne:
                    raise RuntimeError("At compile time, your iterator said "
                            "it had " + str(ne) + " examples total, but at "
                            "runtime it gave us " + str(actual_ne) + ".")
        # end for d
        if self.p_save != None:
            b= open(self.p_save,'a')
            b.write("\tEpochs seen: %d\n" % mm._epochs_seen)
        print("Monitoring step:")
        print("\tEpochs seen: %d" % mm._epochs_seen)
        print("\tBatches seen: %d" % mm._num_batches_seen)
        #print("\tExamples seen: %d" % mm._examples_seen)
        t = time.time() - mm.t0
        #print mm.channels
        for channel_name in self.p_channel:                
            if channel_name in mm.channels:
                channel = mm.channels[channel_name]
                channel.time_record.append(t)
                channel.batch_record.append(mm._num_batches_seen)
                channel.example_record.append(mm._examples_seen)
                channel.epoch_record.append(mm._epochs_seen)
                val = channel.val_shared.get_value()
                # naive hack: ...
                #channel.val_shared.set_value(0)
                channel.val_record.append(val)
                if abs(val) < 1e4:
                    val_str = str(val)
                else:
                    val_str = '%.3e' % val
                print "\t%s: %s" % (channel_name, val_str)
                if self.p_save!=None:
                    b.write("\t%s: %s\n" % (channel_name, val_str)) 
        # clean up        
        if self.p_save!=None:
            b.close() 
        mm._epochs_seen += 1

