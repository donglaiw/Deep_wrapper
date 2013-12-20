# wrapper to create stacked layers
class SetParam():
    def __init__(self):
        pass        
    class param_algo():
        def __init__(self,
            learning_rate, 
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
            seed=[2012, 10, 5]):
            self.learning_rate = learning_rate
            self.cost = cost
            self.batch_size = batch_size
            self.monitoring_batches = monitoring_batches
            self.monitoring_dataset = monitoring_dataset
            self.monitor_iteration_mode = monitor_iteration_mode
            self.termination_criterion = termination_criterion
            self.update_callbacks = update_callbacks
            self.learning_rule = learning_rule
            self.init_momentum = init_momentum
            self.set_batch_size = set_batch_size
            self.train_iteration_mode = train_iteration_mode
            self.batches_per_iter = batches_per_iter
            self.theano_function_mode = theano_function_mode
            self.monitoring_costs = monitoring_costs
            self.seed = seed
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
