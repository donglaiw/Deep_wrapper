import cPickle
import os
import sys
import time
import numpy as np
import Image

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

    def loadFile(self,file_path,which_set,data_id, data_ind):
        if not os.path.exists(file_path):
            print file_path+" : doesn't exist"
            return None
        else:
            # pre-computed
            X_path = file_path + 'X'+which_set+str(data_id)+'.npy'
            Y_path = file_path + 'Y'+which_set+str(data_id)+'.npy'   
             
            if os.path.exists(X_path):
                X = np.load(X_path)
                y = None
                if os.path.exists(Y_path):
                    y = np.load(Y_path)
            else:            
                X, y = self._load_data(file_path, data_id, which_set)
                # default: X=(m,n), m instances of n dimensional feature
                if data_ind!=None:
                    num = X.shape[0]            
                    if max(data_ind)>num:
                        raise('index too big')
                    # print data_ind,num
                    X = X[data_ind, :]

                if not os.path.exists(X_path):
                    np.save(X_path, X)
                    print "save: "+X_path
                if y is not None and not os.path.exists(Y_path):
                    if data_ind!=None:
                        y = y[data_ind]
                    np.save(Y_path, y)         
                    print "save: "+Y_path 
            """
            print y[:10]
            print X[:10,:]                              
            """
            return X,y

    def _load_data(self, file_path, data_id, data_ind, which_set):
        return
    def _load_npy():
        pass

    def label_id2arr(self,y,numclass):
        one_hot = np.zeros((y.shape[0],numclass),dtype='float32')
        for i in xrange(y.shape[0]):
            one_hot[i,y[i]] = 1.
        return one_hot
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
        return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

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
    def __init__(self,which_set,
            base_path = '/data/vision/billf/manifold-learning/DL/Deep_Low/dn/voc/',            
            data_ind = None,   
            options = None,
            axes = ('b', 0, 1, 'c'),                                    
            ):
        self.options = options
        
        if options==None:
            data_id = 0
        else:
            data_id = options['data_id'];
        
        if data_id <=2:
            self.ishape = (17,17,1)
        X, y = self.loadFile(base_path,which_set, data_id, data_ind)            

        
        
        view_converter = DefaultViewConverter(shape = self.ishape, axes=axes)            
        super(Denoise, self).__init__(X=X, y=y, view_converter=view_converter)

    def _load_data(self, file_path, data_id, which_set):        
        import scipy.io     
        if data_id ==0:
            mat = scipy.io.loadmat(file_path+'n_voc_p1.mat')
            # Discard header
            # row = reader.next()
            X = np.matrix.transpose(np.asarray(mat['npss']).astype('float32')/255)
            if which_set != 'test':
                mat = scipy.io.loadmat(file_path+'c_voc_p1.mat')
                y = np.matrix.transpose(np.asarray(mat['pss']).astype('float32')/255)
            else:
                y = None
        elif data_id==1:
            # test for one image
            mat = scipy.io.loadmat(file_path+self.options['data'])
            X = np.asarray(mat['nps']).astype('float32').T/255
            y = np.asarray(mat['ps']).astype('float32').T/255
        elif data_id==2:
            # test for BSD
            mat = scipy.io.loadmat(file_path+self.options['data'])
            im_id  = self.options['im_id']
            tmp_x = self.patchify(mat['Ins'][0][im_id],self.ishape[:2])
            tmp_sz = tmp_x.shape
            X = np.reshape(tmp_x,(tmp_sz[0]*tmp_sz[1],tmp_sz[2]*tmp_sz[3]))
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
    def __init__(self,which_set,numclass,
            base_path = '/data/vision/billf/manifold-learning/DL/Data/icml_2013_emotions',
            data_ind = None,            
            preprocessor = None,            
            fit_preprocessor = False,
            axes = ('b', 0, 1, 'c'),            
            fit_test_preprocessor = False,                        
            flip=0
            ):

        X, y = self.loadFile(base_path,which_set, data_ind)
        # train_index
        if flip:
            X_list_flipLR, X_list_flipUD = self.flipData(X)
            X = X + X_list_flipLR
            y = y + y        
        
        view_converter = DefaultViewConverter(shape=np.append(ishape.shape,ishape.num_channels), axes=axes)
        super(Occ, self).__init__(X=X, y=self.label_id2arr(y,numclass), view_converter=view_converter)

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)
        """
        from pylearn2.datasets.preprocessing import GlobalContrastNormalization
        pre = GlobalContrastNormalization(sqrt_bias = 10,use_std = 1)            
        """                

    def _load_data(path, which_set):     
        pass

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
    def __init__(self,dataset_id):
        #self.data={'train':None,'valid':None,'test':None}
        self.data={}
        self.dataset_id = dataset_id

    def loadData(self,basepath,which_set,data_ind=None,options=None):        
        assert which_set in ['train','valid','test']
        if self.dataset_id==0:            
            self.data[which_set] = Occ(which_set,basepath,data_ind,options)            
        elif self.dataset_id==1:
            self.data[which_set] = ICML_emotion(which_set,basepath,data_ind,options)                
        elif self.dataset_id==2:            
            self.data[which_set] = Denoise(which_set,basepath,data_ind,options)                
