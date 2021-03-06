clc
clear
load rawdata;

image_XxY=double(image_XxY);
image_XY_B=double(image_XY_B);

image_XxY_B=reshape(image_XY_B,size(image_XxY,1),size(image_XxY,2),size(image_XY_B,2));
[m,n,p]=size(image_XxY_B);

%this is used to extend the whole data, as our previous work said
%'Spatial Revising Variational Autoencoder-Based Feature Extraction Method for Hyperspectral Images'
batch=(time_batch-1)/2;
newdata=extenddata(image_XxY_B);

%find the spatial data
squeeze_data=zeros(m*n,time_batch*time_batch,p);
raw_data=zeros(m*n,p);
act_Y_train=[];
indexi=[];
indexj=[];
for i=1:m
    for j=1:n
        image_XxY(i,j,2)=i;
        image_XxY(i,j,3)=j;
    end
end
for i=1:m
    for j=1:n
        sizei=m+i;
        sizej=n+j;
        spice=newdata((sizei-batch):(sizei+batch),(sizej-batch):(sizej+batch),:);
        spice=reshape(spice,1,time_batch*time_batch,p);
        squeeze_data((i-1)*n+j,:,:)=spice;
        raw_data((i-1)*n+j,:)=reshape(newdata(sizei,sizej,:),1,p);
        act_Y_train=[act_Y_train;image_XxY(i,j,1)];
        indexi=[indexi;image_XxY(i,j,2)];
        indexj=[indexj;image_XxY(i,j,3)];
    end
end
%remove samples with 0 label
index=find(act_Y_train~=0);
squeeze_data_delete0=squeeze_data(index,:,:);
raw_data_delete0=raw_data(index,:);
act_Y_train=act_Y_train(index);
indexi=indexi(index);
indexj=indexj(index);

X_LSTM=squeeze_data_delete0;
X_VAE=raw_data_delete0;

act_Y_train=single(act_Y_train);
X_VAE=single(X_VAE);
X_LSTM=single(X_LSTM);
indexi=single(indexi);
indexj=single(indexj);
image_XxY_B=single(image_XxY_B);
image_XxY=single(image_XxY);

save 92av3c_spatialdata X_LSTM indexi indexj
