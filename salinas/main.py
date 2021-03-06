import keras
import pylab
import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras import layers
from keras import backend as K
import matplotlib.pyplot as plt 
from keras_layer_normalization import LayerNormalization

indexi=sio.loadmat('salinas_spatialdata.mat')['indexi']
indexj=sio.loadmat('salinas_spatialdata.mat')['indexj']
X_spatial=sio.loadmat('salinas_spatialdata.mat')['X_LSTM']

windowsize=9
X_spatial=np.reshape(X_spatial,[X_spatial.shape[0],windowsize,windowsize,X_spatial.shape[2]])
X_spatial=X_spatial*2-1#[-1,1] normalization

nei_number=4#the band neighbor

#this part is used to constract the spectral reflectance variance
flow_X_spatial1=np.zeros(shape=(X_spatial.shape[0],X_spatial.shape[1],X_spatial.shape[2],nei_number,X_spatial.shape[3]))
flow_X_spatial2=np.zeros(shape=(X_spatial.shape[0],X_spatial.shape[1],X_spatial.shape[2],nei_number*X_spatial.shape[3]))
for i in range(X_spatial.shape[3]):
    original=X_spatial[:,:,:,i].reshape([X_spatial.shape[0],X_spatial.shape[1],X_spatial.shape[2]])
    original_dim=np.zeros(shape=(original.shape[0],original.shape[1],original.shape[2],nei_number+1))
    for j in range(nei_number+1):
        original_dim[:,:,:,j]=original
    neiindex=np.arange(i-int(nei_number/2),i+int(nei_number/2)+1,1)
    low_neiindex=neiindex*0
    high_neiindex=neiindex*0+X_spatial.shape[3]-1
    neiindex=neiindex*(neiindex>=low_neiindex)+low_neiindex*(neiindex<low_neiindex)
    neiindex=neiindex*(neiindex<=high_neiindex)+high_neiindex*(neiindex>high_neiindex)
    neiband=X_spatial[:,:,:,neiindex]
    neiband=neiband-original_dim
    neiband=np.delete(neiband,int(nei_number/2),axis=3)
    flow_X_spatial1[:,:,:,:,i]=neiband
    flow_X_spatial2[:,:,:,np.arange(nei_number*i,nei_number*(i+1))]=neiband
    
#do shuffle
shuffle=1
if shuffle==1:
    index=np.arange(X_spatial.shape[0])
    np.random.seed(1)
    np.random.shuffle(index)
    X_spatial=X_spatial[index]
    flow_X_spatial1=flow_X_spatial1[index]
    flow_X_spatial2=flow_X_spatial2[index]
    indexi=indexi[index]
    indexj=indexj[index]

#parameters
crom_parameter=0.3
X_spatial_train=X_spatial[0:int(crom_parameter*X_spatial.shape[0]),:,:,:]
flow_X_spatial1_train=flow_X_spatial1[0:int(crom_parameter*flow_X_spatial1.shape[0]),:,:,:,:]
flow_X_spatial2_train=flow_X_spatial2[0:int(crom_parameter*flow_X_spatial2.shape[0]),:,:,:]
iterations=100000
latent_vector=100
latent_dim=30
batch_size=64
lr=1e-3

kernel_regularizer=None

def sequeeze(x):
    x=K.squeeze(x,1)
    return x
def middle(x):
    x=x[:,4,4,:]
    return x
def interpolating(x):
    u=K.random_uniform((batch_size,)+(1,)*(K.ndim(x[0])-1))
    return x[0]*u+x[1]*(1-u)
############################   generator  #####################################
generator_input=keras.Input(shape=(latent_vector,))
x_stream1=layers.Dense(1*1*512,kernel_regularizer=kernel_regularizer)(generator_input)
x_stream1=layers.Reshape((1,1,512))(x_stream1)

x_stream2=layers.Dense(1*1*1*512,kernel_regularizer=kernel_regularizer)(generator_input)
x_stream2=layers.Reshape((1,1,1,512))(x_stream2)

x_stream1=layers.Conv2DTranspose(256,(5,5),strides=(1,1),padding='valid',kernel_regularizer=kernel_regularizer)(x_stream1)
x_stream1=layers.LeakyReLU()(x_stream1)
x_stream1=layers.BatchNormalization()(x_stream1)

x_stream1=layers.Conv2DTranspose(flow_X_spatial1.shape[4],(5,5),strides=(1,1),padding='valid',kernel_regularizer=kernel_regularizer)(x_stream1)
x_stream1=layers.LeakyReLU()(x_stream1)
x_stream1=layers.BatchNormalization()(x_stream1)

x_stream2=layers.Conv3DTranspose(256,(5,5,2),strides=(1,1,2),padding='valid',kernel_regularizer=kernel_regularizer)(x_stream2)
x_stream2=layers.LeakyReLU()(x_stream2)
x_stream2=layers.BatchNormalization()(x_stream2)

x_stream2=layers.Conv3DTranspose(flow_X_spatial1.shape[4],(5,5,2),strides=(1,1,2),padding='valid',kernel_regularizer=kernel_regularizer)(x_stream2)
x_stream2=layers.LeakyReLU()(x_stream2)
x_stream2=layers.BatchNormalization()(x_stream2)

generator=keras.models.Model(generator_input,[x_stream1,x_stream2])
generator.summary()
######################   discriminator    #####################################
discriminator_input1=layers.Input(shape=(windowsize,windowsize,flow_X_spatial1.shape[4]))
discriminator_input2=layers.Input(shape=(windowsize,windowsize,flow_X_spatial1.shape[3],flow_X_spatial1.shape[4]))

discriminator1=layers.Conv2D(256,(5,5),strides=(1,1),padding='valid',kernel_regularizer=kernel_regularizer)(discriminator_input1)
discriminator1=layers.LeakyReLU()(discriminator1)
discriminator1=LayerNormalization()(discriminator1)

discriminator1=layers.Conv2D(latent_dim,(5,5),strides=(1,1),padding='valid',kernel_regularizer=kernel_regularizer)(discriminator1)
discriminator1=layers.LeakyReLU()(discriminator1)
discriminator1=LayerNormalization()(discriminator1)

feature1=layers.AveragePooling2D((1,1))(discriminator1)
feature1=layers.Lambda(sequeeze)(feature1)
feature1=layers.Lambda(sequeeze)(feature1)
discriminator1=layers.Flatten()(discriminator1)

discriminator2=layers.Conv3D(256,(5,5,2),strides=(1,1,2),padding='valid',kernel_regularizer=kernel_regularizer)(discriminator_input2)
discriminator2=layers.LeakyReLU()(discriminator2)
discriminator2=LayerNormalization()(discriminator2)

discriminator2=layers.Conv3D(latent_dim,(5,5,2),strides=(1,1,2),padding='valid',kernel_regularizer=kernel_regularizer)(discriminator2)
discriminator2=layers.LeakyReLU()(discriminator2)
discriminator2=LayerNormalization()(discriminator2)

feature2=layers.AveragePooling3D((1,1,1))(discriminator2)
feature2=layers.Lambda(sequeeze)(feature2)
feature2=layers.Lambda(sequeeze)(feature2)
feature2=layers.Lambda(sequeeze)(feature2)
discriminator2=layers.Flatten()(discriminator2)

finalfeature=layers.concatenate([feature1,feature2])

discriminator=layers.concatenate([discriminator1,discriminator2])

DDiscriminator=keras.models.Model([discriminator_input1,discriminator_input2],discriminator)
DDiscriminator.summary()
feature_extractor=keras.models.Model([discriminator_input1,discriminator_input2],finalfeature)
feature_extractor.summary()
###################################module1#####################################
x_true_1=layers.Input(shape=(windowsize,windowsize,flow_X_spatial1.shape[4]))
x_true1=layers.Reshape((windowsize,windowsize,flow_X_spatial1.shape[4]))(x_true_1)

x_true_2=layers.Input(shape=(windowsize,windowsize,flow_X_spatial1.shape[3],flow_X_spatial1.shape[4]))
x_true2=x_true_2

z_fake=layers.Input(shape=(latent_vector,))
x_fake1,x_fake2=generator(z_fake)

x_inter1=layers.Lambda(interpolating)([x_true1,x_fake1])
x_inter2=layers.Lambda(interpolating)([x_true2,x_fake2])

x_real_score=DDiscriminator([x_true1,x_true2])
x_fake_score=DDiscriminator([x_fake1,x_fake2])
x_inter_score=DDiscriminator([x_inter1,x_inter2])

grads1=K.gradients(x_inter_score,x_inter1)[0]
grads2=K.gradients(x_inter_score,x_inter2)[0]

grad_norms1=tf.norm(grads1,ord='euclidean')
grad_norms2=tf.norm(grads2,ord='euclidean')

discriminator_train_model=keras.models.Model([x_true_1, x_true_2, z_fake],[x_real_score,x_fake_score,x_inter_score])
D_loss = tf.reduce_mean(x_fake_score)-tf.reduce_mean(x_real_score)+10*tf.reduce_mean((grad_norms1+grad_norms2-2)**2)
discriminator_train_model.add_loss(D_loss)
discriminator_train_model.compile(optimizer=keras.optimizers.adam(lr=lr))
discriminator_train_model.summary()
###################################module2#####################################
DDiscriminator.trainable = False
x_fake_score_two=DDiscriminator([x_fake1,x_fake2])
generator_train_model = keras.models.Model(z_fake, x_fake_score_two)
generator_train_model.add_loss(-tf.reduce_mean(x_fake_score_two))
generator_train_model.compile(optimizer=keras.optimizers.adam(lr=lr))
generator_train_model.summary()
###################################run#########################################
discriminator_number=1#the number of discriminator running
generator_number=1#the number of generator running

d_final_loss=np.zeros(shape=iterations*discriminator_number)      
g_final_loss=np.zeros(shape=iterations*generator_number)  

for i in range(iterations):
    for j in range(discriminator_number):
        index = [ind for ind in range(int(crom_parameter*X_spatial.shape[0]))]
        np.random.shuffle(index)
        x_train1_change=X_spatial_train[index,:,:,:]
        real_batch_X1=x_train1_change[0:batch_size,:,:,:]
        x_train21_change=flow_X_spatial1_train[index,:,:,:,:]
        real_batch_X21=x_train21_change[0:batch_size,:,:,:,:]
        x_train22_change=flow_X_spatial2_train[index,:,:,:]
        real_batch_X22=x_train22_change[0:batch_size,:,:,:]
        random_latent_vectors=np.random.random(size=(batch_size,latent_vector))*2-1
        D_loss = discriminator_train_model.train_on_batch([real_batch_X1, real_batch_X21, random_latent_vectors],None)
        d_final_loss[i*discriminator_number+j]=D_loss
    for j in range(generator_number):
        random_latent_vectors=np.random.random(size=(batch_size,latent_vector))*2-1
        G_loss = generator_train_model.train_on_batch(random_latent_vectors, None)
        g_final_loss[i*generator_number+j]=G_loss
     
    #the training is time-consuming, so I save weights per 50 iterations
    if i%50==0:
        discriminator_train_model.save_weights('discriminator_train_model'+str(i)+'.h5')
        generator_train_model.save_weights('generator_train_model'+str(i)+'.h5')
        
    #show the loss
    if i%1==0:
        print('iter: %s, d_loss: %s, g_loss: %s' % (i, D_loss, G_loss))
        
    #show the loss line per 10 iterations
    if i%10==0:
        plt.plot(d_final_loss[0:i*discriminator_number+j])
        pylab.show()
        plt.plot(g_final_loss[0:i*generator_number+j])
        pylab.show()
        
    #do the testing per 500 iterations to see the training result
    if i%500==0:
        finalfeature=feature_extractor.predict([np.reshape(X_spatial,[X_spatial.shape[0],windowsize,windowsize,X_spatial.shape[2]]),flow_X_spatial1])
        
        #save them
        name='svmdata'+str(i)+'.mat'
        sio.savemat(name, {'finalfeature':finalfeature,#this is the latent embeddings
                           'cross_param':crom_parameter,
                           'iteration':i,
                           'indexi':indexi,
                           'indexj':indexj,
                           'd_final_loss':d_final_loss,
                           'g_final_loss':g_final_loss,
                           'G_loss':G_loss,
                           'D_loss':D_loss})
    
