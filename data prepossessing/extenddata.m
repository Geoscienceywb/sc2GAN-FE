function data=extenddata(image_XxY_B)
[m,n,p]=size(image_XxY_B);
data=zeros(m*3,n*3,p);

data(1:m,1:n,1:p)=flip(flip(image_XxY_B,2),1);
data((1*m+1):(2*m),1:n,1:p)=flip(image_XxY_B,2);
data((2*m+1):(3*m),1:n,1:p)=flip(flip(image_XxY_B,2),1);

data(1:m,(1*n+1):(2*n),1:p)=flip(image_XxY_B,1);
data((1*m+1):(2*m),(1*n+1):(2*n),1:p)=image_XxY_B;
data((2*m+1):(3*m),(1*n+1):(2*n),1:p)=flip(image_XxY_B,1);

data(1:m,(2*n+1):(3*n),1:p)=flip(flip(image_XxY_B,2),1);
data((1*m+1):(2*m),(2*n+1):(3*n),1:p)=flip(image_XxY_B,2);
data((2*m+1):(3*m),(2*n+1):(3*n),1:p)=flip(flip(image_XxY_B,2),1);