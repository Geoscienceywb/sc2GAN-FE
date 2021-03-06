function SVMused(index)
namee=['svmdata',num2str(index)];
load(namee)
load 92av3c_145x145_200
%load paviaU_300x145_103
%load salinas_512x217_204
%load DFC_900x600_50
[m,n,p]=size(image_XxY_B);
generated_feature=finalfeature;%this is what we obtained
[mm,pp]=size(generated_feature);
new_combined_data=zeros(size(image_XxY_B,1),size(image_XxY_B,2),pp);
for i=1:size(generated_feature,1)
    spice=generated_feature(i,:);
    index_x=indexi(i);
    index_y=indexj(i);
    new_combined_data(index_x,index_y,:)=spice;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nclasses=max(max(image_XxY));
OA_total=zeros(1,10);
AA_total=zeros(1,10);
kappa_total=zeros(1,10);
cs_total=zeros(max(max(image_XxY)),10);
for judge=1:10
    [TrainSamples,TrainLabels,TestSamples,TestLabels,TrainIndex,TestIndex]=SvmData(new_combined_data,image_XxY,10,judge,1);
    %do SVMcg selection
    [bestacc,bestc,bestg]=SVMcg(TrainLabels, TrainSamples,-4,4,-4,4,5,0.5,0.5,0.9);
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -b 1'];
    model=svmtrain(TrainLabels,TrainSamples,cmd);
    [predict_label,accuracy,prob_estimates]=svmpredict(TestLabels,TestSamples,model,'-b 1');
    Predict_label=zeros(m,n);
    Predict_label(TrainIndex)=TrainLabels;
    Predict_label(TestIndex)=predict_label;
    num=zeros(1,nclasses);
    summ=zeros(1,nclasses);
    total=0;
    for i=1:length(predict_label)
        summ(1,TestLabels(i))=summ(1,TestLabels(i))+1;
        if predict_label(i)==TestLabels(i,1)
            num(1,predict_label(i))=num(1,predict_label(i))+1;
            total=total+1;
        end
    end
    for i=1:nclasses
        css(i,1)=num(1,i)/summ(1,i);
    end
    OA=total/sum(summ);
    AA=mean(css);
    OA_total(1,judge)=OA;
    AA_total(1,judge)=AA;
    cs_total(:,judge)=css;
    orilabel=TestLabels;
    label=predict_label;
    decnum=zeros(1,max(orilabel));
    yynum=zeros(1,max(orilabel));
    for i=1:max(orilabel)
        for j=1:size(predict_label,1)
            if label(j)==i
                decnum(i)=decnum(i)+1;
            end
            if orilabel(j)==i
                yynum(i)=yynum(i)+1;
            end
        end
    end
    total=0;
    for i=1:max(orilabel)
        total=total+decnum(i)*yynum(i);
    end
    total=total/(size(predict_label,1)*size(predict_label,1));
    kappa=(OA-total)/(1-total);
    kappa_total(1,judge)=kappa;
end
final_OA=mean(OA_total);
std_OA=sqrt(sum((OA_total(:,1)-mean(OA_total)).^2)/length(OA_total));
final_AA=mean(AA_total);
std_AA=sqrt(sum((AA_total(:,1)-mean(AA_total)).^2)/length(AA_total));
final_kappa=mean(kappa_total);
std_kappa=sqrt(sum((kappa_total(:,1)-mean(kappa_total)).^2)/length(kappa_total));
for ii=1:max(max(image_XxY))
    cs_i=cs_total(ii,:);
    final_cs(ii)=mean(cs_i);
    std_cs(ii)=sqrt(sum((cs_i(:,1)-mean(cs_i)).^2)/length(cs_i));
end
svmresult=['proposed','-',num2str(cross_param*10),'-',num2str(pp),'-',num2str(iteration),'iteration'];
save(svmresult,'final_OA','std_OA','final_AA','std_AA','final_cs','std_cs','Predict_label','final_kappa','std_kappa','cmd','G_loss','D_loss','iteration')