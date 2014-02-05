import cPickle
import os
import sys
import time
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.preprocessing import GlobalContrastNormalization
from pylearn2.utils.string_utils import preprocess


"""
parent class with preprocessor
"""
class DataIO(DenseDesignMatrix):
    def __init__(self,X, y, view_converter):        
        super(DataIO, self).__init__(X=X, y=y, view_converter=view_converter)
    """
    import numpy as np
    X = np.load('data/train/Xtrain0.npy')    
    y = np.load('data/train/Ytrain0.npy')    
    """
    def setup(self,options):
        self.im_id = 0
        self.mat_id = 0
        self.dname = ''
        self.data_id = 0
        self.crop_y = None
        self.ishape = None
        self.axes = ('b', 0, 1, 'c')
        self.pre_id = 0

        if options!= None:
            if 'ishape' in options:
                self.ishape = options['ishape']
            if 'mat_id' in options:
                self.mat_id = options['mat_id']
            if 'data_id' in options:
                self.data_id = options['data_id']
            if 'data' in options:
                self.dname  = options['data']
            if 'im_id' in options:
                self.im_id  = options['im_id']
            if 'axes' in options:
                self.axes  = options['axes']
            if 'pre_id' in options:
                self.pre_id = options['pre_id']
            if 'crop_y' in options:
                self.crop_y = options['crop_y']
                #print "load c_y"

    def loadFile(self,file_path,which_set, data_ind):
        if not os.path.exists(file_path):
            print file_path+" : doesn't exist"
            return None
        else:
            # pre-computed
            dname = self.dname[:self.dname.find('.')]
            X_path = file_path + 'X'+which_set+'_'+dname+'_'+str(self.data_id)+'_'+str(self.im_id)+'.npy'
            Y_path = file_path + 'Y'+which_set+'_'+dname+'_'+str(self.data_id)+'_'+str(self.im_id)+'.npy'   
            #print X_path 
            if os.path.exists(X_path):
                X = np.load(X_path)
                y = None
                if os.path.exists(Y_path):
                    y = np.load(Y_path)
            else:            
                #print "do it"
                X, y = self._load_data(file_path,  which_set)
                #print "done"
                # default: X=(m,n), m instances of n dimensional feature
                #print "start_save",len(data_ind),X.shape,y.shape
                if data_ind!=None:
                    num = X.shape[0]            
                    if max(data_ind)>num:
                        raise('index too big')
                    # print data_ind,num
                    X = X[data_ind, :]
                #print "after cut",len(data_ind),X.shape
                if not os.path.exists(X_path):
                    np.save(X_path, X)
                    print "save: "+X_path
                if y is not None:
                    if data_ind!=None:
                        y = y[data_ind]
                    if self.crop_y!=None:
                        print "crop y"
                        y = y[:,self.crop_y]

                    np.save(Y_path, y)         
                    print "save: "+Y_path 
                    #print self.crop_y
            """
            print y[:10]
            print X[:10,:]                              
            """
            return X,y

    def _load_data(self, file_path, data_ind, which_set):
        raise NotImplementedError(str(type(self)) + " does not implement: _load_data().")

    def label_id2arr(self,y,numclass):
        one_hot = np.zeros((y.shape[0],numclass),dtype='float32')
        for i in xrange(y.shape[0]):
            one_hot[i,y[i]] = 1.
        return one_hot
    def patchify3(self,img, patch_shape):
        assert(len(img.shape)==3)
        patch_shape = patch_shape[:2]
        out = self.patchify(img[:,:,0],patch_shape)
        for i in range(1,img.shape[-1]):
            out = np.hstack((out,(self.patchify(img[:,:,i],patch_shape))))
        return out
    def patchify(self,img, patch_shape):
        img = np.ascontiguousarray(img)  # won't make a copy if not needed
        X, Y = img.shape
        x, y = patch_shape
        shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
        # The right strides can be thought by:
        # 1) Thinking of `img` as a chunk of memory in C order
        # 2) Asking how many items through that chunk of memory are needed when indices
        #    i,j,k,l are incremented by one
        strides = img.itemsize*np.array([Y, 1, Y, 1])
        tmp_x = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
        tmp_sz = tmp_x.shape
        return np.reshape(tmp_x,(tmp_sz[0]*tmp_sz[1],tmp_sz[2]*tmp_sz[3]))

    def flipData(self, sampleList, sizeImg = [48, 48]):
            # flip the image set from left to right, and upside down
            print "flip the images"
            sampleList_LR = []
            sampleList_UD = []
            for i in range(len(sampleList)):
                curSampleVector = sampleList[i]
                singleImg = np.asarray(curSampleVector).astype('uint8').reshape((sizeImg[0],sizeImg[1]))
                #singleImg = singleImg.reshape((sizeImg[0],sizeImg[1]))
                singleImg_ud = np.flipud(singleImg)
                singleImg_lr = np.fliplr(singleImg)
                sampleList_UD.append(list(singleImg_ud.reshape((sizeImg[0]*sizeImg[1]))))
                sampleList_LR.append(list(singleImg_lr.reshape((sizeImg[0]*sizeImg[1]))))
            return sampleList_LR, sampleList_UD

class Denoise(DataIO):    
    def __init__(self,
        options,
        base_path,
        which_set,
        data_ind = None):
        self.setup(options)

        if self.data_id <=2:
            self.ishape = (17,17,1)
        X, y = self.loadFile(base_path,which_set, data_ind)
        view_converter = DefaultViewConverter(shape = self.ishape, axes=self.axes)            
        super(Denoise, self).__init__(X=X, y=y, view_converter=view_converter)

    def _load_data(self, file_path, which_set):        
        import scipy.io     
        if self.data_id ==0:
            X = np.zeros(( 0, np.prod(self.ishape)), dtype = np.float32)
            for i in self.mat_id:
                mat = scipy.io.loadmat(file_path+'n_voc_p'+str(i)+'.mat')
                #print i,mat['npss'].shape
                X = np.vstack((X,(np.asarray(mat['npss']).T.astype('float32')/255-0.5)/0.2))
            if which_set != 'test':
                y = np.zeros(( 0, np.prod(self.ishape)), dtype = np.float32)
                for i in self.mat_id:
                    mat = scipy.io.loadmat(file_path+'c_voc_p'+str(i)+'.mat')
                    #print i,mat['pss'].shape
                    y = np.vstack((y,(np.asarray(mat['pss']).T.astype('float32')/255-0.5)/0.2))
        elif self.data_id==1:
            # test for one image
            mat = scipy.io.loadmat(file_path+self.dname)
            X = (np.asarray(mat['nps']).astype('float32').T/255-0.5)/0.2
            y = (np.asarray(mat['ps']).astype('float32').T/255-0.5)/0.2
        elif self.data_id==2:
            # test for BSD
            mat = scipy.io.loadmat(file_path+self.dname)
            mat = (np.asarray(mat['Ins'][0][self.im_id]).astype('float32')/255-0.5)/0.2
            X = self.patchify(mat,self.ishape[:2])
            y = None

            """
            # out of gpu memory
            X = np.zeros(( 0, np.prod(self.ishape)), dtype = np.float32)
            y = np.zeros(( 0, np.prod(self.ishape)), dtype = np.float32)
            for i in range(len(mat['Is'][0])):
                tmp_x = self.patchify(mat['Ins'][0][0],self.ishape[:2])
                tmp_sz = tmp_x.shape
                X = np.vstack((X,np.reshape(tmp_x,(tmp_sz[0]*tmp_sz[1],tmp_sz[2]*tmp_sz[3]))))
                tmp_x = self.patchify(mat['Is'][0][0],self.ishape[:2])
                tmp_sz = tmp_x.shape
                y = np.vstack((X,np.reshape(tmp_x,(tmp_sz[0]*tmp_sz[1],tmp_sz[2]*tmp_sz[3]))))
            """
        return X, y

class Occ(DataIO):    
    def __init__(self,
        options,
        base_path,
        which_set,
        data_ind = None):
        self.setup(options)
        #print "yo:",self.data_id,options
        X, y = self.loadFile(base_path,which_set, data_ind)            
        view_converter = DefaultViewConverter(shape = self.ishape, axes=self.axes)
        super(Occ, self).__init__(X=X, y=y, view_converter=view_converter)


    def _load_data(self, file_path, which_set):
        import scipy.io             
        if self.data_id == -1:
            # 2 class
            varname = self.dname[:self.dname.find('.')]
            mat = scipy.io.loadmat(file_path+self.dname)
            X = np.asarray(mat[varname][1:]).astype('float32').T
            if which_set != 'test':                
                y = np.asarray(mat[varname][0]).astype('float32')
                y[y!=150]=0
                y[y==150]=1
                #print 'test_y: ',y[:10]
                y = self.label_id2arr(y,2);
                #print 'test_y2: ',y[0]
            else:
                y = None
        elif self.data_id ==0:
            # 151 classes
            varname = self.dname[:self.dname.find('.')]
            mat = scipy.io.loadmat(file_path+self.dname)
            X = np.asarray(mat[varname][1:]).astype('float32').T
            if which_set != 'test':                
                y = np.asarray(mat[varname][0]).astype('float32')
                #print 'test_y: ',y[:10]
                y = self.label_id2arr(y,151);
                #print 'test_y2: ',y[0]
            else:
                y = None
            #print self.p_data
        elif self.data_id==1:
            # test for bench
            mat = scipy.io.loadmat(file_path+self.dname)
            X = np.asarray(mat['test_im'][1:]).astype('float32').T
            y = np.asarray(mat['test_im'][0]).astype('float32')
        elif self.data_id==2:
            # test for BSD
            pshape = (35,35)
            mat = scipy.io.loadmat(file_path+self.dname)
            if mat['Is2'][0][self.im_id].ndim == 3:
                X = self.patchify3(mat['Is2'][0][self.im_id],pshape)
            else: 
                X = self.patchify(mat['Is2'][0][self.im_id],pshape)
            #y = self.label_id2arr(np.ones((X.shape[0],1),dtype='float32'),151)
            X = X.astype('float32')
            y = None
        elif self.data_id ==3:
            # regression
            mat = scipy.io.loadmat(file_path+self.dname)
            X = np.asarray(mat[self.dname[:-4]][1:]).astype('float32').T
            if which_set != 'test':                
                mat = scipy.io.loadmat(file_path+'train_bd.mat')
                y = np.asarray(mat['train_bd']).astype('float32').T
                #print 'test_y: ',y[:10]
                #print 'test_y2: ',y[0]
                #print "data_id 3:",y.shape
            else:
                y = None
        elif self.data_id ==4:
            # regression
            mat = scipy.io.loadmat(file_path+self.dname)
            X = np.asarray(mat['train_im']).astype('float32').T
            if which_set != 'test':                
                y = np.asarray(mat['train_bd']).astype('float32').T
                #print 'test_y: ',y[:10]
                #print 'test_y2: ',y[0]
                #print "data_id 3:",y.shape
            else:
                y = None
        elif self.data_id ==5:
            # regression
            mat = scipy.io.loadmat(file_path+'decaf_5_'+str(self.mat_id[0])+self.dname[self.dname.rfind('_'):])
            tmp_X = mat['fmap'][0]
            dim = tmp_X[0].shape
            X = np.zeros(( 0, dim[-1]), dtype = np.float32)
            for im_X in tmp_X:
                dim = im_X.shape
                X = np.vstack((X,np.reshape(np.asarray(im_X).astype('float32'),(np.prod(dim[:2]),dim[-1]))))
            if True or which_set != 'test':                
                tmp_y = mat['fgt'][0]
                y = np.zeros(( 0, 1), dtype = np.float32)
                for im_y in tmp_y:
                    y = np.vstack((y,np.reshape(np.asarray(im_y).astype('float32'),(im_y.size,1))))
                #print 'test_y: ',y[:10]
                #print 'test_y2: ',y[0]
                #print "data_id 3:",y.shape
            else:
                y = None       
        elif self.data_id ==6:
            # regression
            mat = scipy.io.loadmat(file_path+self.dname)
            #print file_path+self.dname
            X = np.asarray(mat['mat_x']).astype('float32').T
            y = np.asarray(mat['mat_y']).astype('float32').T
            
        if self.pre_id==1:
            X = (X/255-0.5)/0.2
            if self.data_id ==4 and y != None:
                y = (y/255-0.5)/0.2

        return X, y

class ICML_emotion(DataIO):
    """
    A Pylearn2 Dataset class for accessing the data for the
    facial expression recognition Kaggle contest for the ICML
    2013 workshop on representation learning.
    """
    def __init__(self,which_set,numclass,
            base_path = '/data/vision/billf/manifold-learning/DL/Data/icml_2013_emotions',
            start = 0,
            stop = -1,
            options = [0],
            axes = ('b', 0, 1, 'c'),            
            fit_test_preprocessor = False,                                    
            ):
        files = {'train': 'train.csv', 'public_test' : 'test.csv'}
        try:
            file_path = files[which_set]
        except KeyError:
            raise ValueError("Unrecognized dataset name: " + which_set)
        
        X, y = self.loadFile(base_path + '/' + file_path, start,stop)
        # train_index
        if flip:
            X_list_flipLR, X_list_flipUD = self.flipData(X)
            X = X + X_list_flipLR
            y = y + y    

        view_converter = DefaultViewConverter(shape=(48,48,1), axes=axes)
        super(ICML_emotion, self).__init__(X=X, y=self.label_id2arr(y,numclass), view_converter=view_converter)
                
        if options[0] == 1:
            fit_preprocessor = False
            from pylearn2.datasets.preprocessing import GlobalContrastNormalization
            preprocessor = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)            
            preprocessor.apply(self, can_fit=fit_preprocessor)

    def _load_data(self, path, expect_labels):
        assert path.endswith('.csv')
        # Convert the .csv file to numpy
        csv_file = open(path, 'r')
        import csv
        reader = csv.reader(csv_file)
        # Discard header
        row = reader.next()
        y_list = []
        X_list = []
        for row in reader:
            if expect_labels:
                y_str, X_row_str = row
                y = int(y_str)
                y_list.append(y)
            else:
                X_row_str ,= row
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            X_list.append(X_row)

        X = np.asarray(X_list).astype('float32')
        if expect_labels:
            y = np.asarray(y_list)
        else:
            y = None
        return X, y
            
class DBL_Data():
    def __init__(self):
        #self.data={'train':None,'valid':None,'test':None}
        self.data={}

    def loadData(self,p_data,basepath,which_set,data_ind=None):
        assert which_set in ['train','valid','test']
        ds_id = p_data['ds_id']
        if ds_id==0:
            self.data[which_set] = Occ(p_data,basepath,which_set,data_ind)            
        elif ds_id==1:
            self.data[which_set] = ICML_emotion(p_data,basepath,which_set,data_ind)                
        elif ds_id==2:
            self.data[which_set] = Denoise(p_data,basepath,which_set,data_ind)
        
