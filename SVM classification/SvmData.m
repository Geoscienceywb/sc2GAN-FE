function [TrainSamples,TrainLabels,TestSamples,TestLabels,index_train,index_test]=SvmData(image_XxY_B,image_XxY,group,serial,preprocess)
Species=[1:1:max(max(image_XxY))];
[X,Y,B]=size(image_XxY_B);
image_XY_B=reshape(image_XxY_B,X*Y,B);
image_XY_=reshape(image_XxY,X*Y,1);
Labels=[];
index=[];
rand('seed',1);
for i=1:length(Species)
    c=length(Labels);
    a_=find(image_XY_==Species(i));
    randIndex=randperm(size(a_,1));%打乱数组排列顺序
    a_=a_(randIndex,:);
    b=length(a_);
    Labels(1+c:b+c)=i;
    index(1+c:b+c)=a_;
    Samples(1:B,1+c:b+c)=image_XY_B(a_,1:B)';
end
pq=length(Labels);
%because in our last work, one reviewer said the number of samples in cluster is too small, we can choose more '9' samples, so we do like this
%from 4316 to 4335 are '9' samples
%you can also use
%pTrain=[serial:group:pq];
pTrain=[serial:group:4315,4316:(group-5):4335,4336:group:pq];%this is only for 92av3c
qTest=1:pq;
qTest(pTrain)=[];

TrainLabels=Labels(pTrain);
index_train=index(pTrain);
TestLabels=Labels(qTest);
index_test=index(qTest);
Samples1(1:B,:)=Samples(:,pTrain);
Samples2(1:B,:)=Samples(:,qTest);

if preprocess==1
    Upper=1;
    Lower=-1;
    MaxV=max(Samples1);
    MinV=min(Samples1);
    [R,C]= size(Samples1);
    TrainSamples=(Samples1-ones(R,1)*MinV).*(ones(R,1)*((Upper-Lower)*ones(1,C)./(MaxV-MinV)))+Lower;
    MaxV=max(Samples2);
    MinV=min(Samples2);
    [R,C]= size(Samples2);
    TestSamples=(Samples2-ones(R,1)*MinV).*(ones(R,1)*((Upper-Lower)*ones(1,C)./(MaxV-MinV)))+Lower;
end

for i=1:length(Species)
    TrainPixel(i)=length(find(TrainLabels==i));
    TestPixel(i)=length(find(TestLabels==i));
end
AllPixel=[Species;TrainPixel;TestPixel;TrainPixel+TestPixel];
disp('Species; TrainPixel; TestPixel; TrainPixel+TestPixel')
disp(AllPixel)
disp('-------------------------------------------------------------------------zmSvmData')
TrainSamples=TrainSamples';
TrainLabels=TrainLabels';
TestSamples=TestSamples';
TestLabels=TestLabels';