# unit test of decaf funcitons
from decaf.layers.cpp import wrapper
import numpy as np
# foward
_ksize = 2
_stride = 1
def test_forward():
    # same as matlab
    padded_data = np.reshape(range(48),(1,4,4,3)).astype('float32')
    sz = padded_data.shape
    col_data = np.zeros((1,(sz[1]-_ksize)/_stride+1,(sz[2]-_ksize)/_stride+1,sz[3]*_ksize*_ksize),'float32')
    wrapper.im2col_forward(padded_data, col_data,_ksize, _stride)
    print col_data

def test_backward():
    # try to assign back
    col_data = np.reshape(range(27),(1,3,3,3)).astype('float32')
    sz = col_data.shape
    col_data = np.zeros((1,sz[1],sz[2],sz[3]*_ksize*_ksize),'float32')
    padded_data = np.zeros((1,4,4,3),'float32')
    wrapper.im2col_backward(padded_data, col_data,_ksize, _stride)
    print padded_data


#col_data = np.reshape(range(12),(1,2,2,3)).astype('float32')
col_data = np.reshape(range(108),(1,3,3,3*4)).astype('float32')
#col_data = np.reshape(range(27),(1,3,3,3)).astype('float32')
sz = col_data.shape
padded_data = np.zeros((1,4,4,3),'float32')
wrapper.im2col_backward(padded_data, col_data,_ksize, _stride)
print padded_data

