import scipy.io as sio
from sklearn import preprocessing
THEANO_FLAGS="device=gpu0,floatX=float32,optimizer=None"

#input the raw data, you should change the name as you like
mat='92av3c_145x145_200.mat'
#mat='paviaU_300x145_103.mat'
#mat='salinas_512x217_204.mat'
#mat='DFC_900x600_50.mat'

data=sio.loadmat(mat)
image_XxY_B=data['image_XxY_B'].astype('float32')#the hyperspectral data
image_XxY=data['image_XxY']#the label

image_XY_B=image_XxY_B.reshape(image_XxY_B.shape[0]*image_XxY_B.shape[1],image_XxY_B.shape[2],order = 'F')

#prepossessing
min_max_scaler=preprocessing.MinMaxScaler()
image_XY_B=min_max_scaler.fit_transform(image_XY_B)

time_batch=9#this is the sliding window size, time_batch*time_batch
sio.savemat('rawdata.mat',{'image_XY_B':image_XY_B,'image_XxY':image_XxY,'time_batch':time_batch})