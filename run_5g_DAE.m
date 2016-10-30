tic
%--- Training AutoEncoder ----%
clc;
clear all;
close all;

%----------Load Training Data-----------------%
[parentdir,~,~]=fileparts(pwd);
global traindata
[traindata] = textread(strcat(parentdir,'/Data/digitstrain.txt'),'','delimiter',',');
nSamples = size(traindata,1);

%----------Load Validation Data-----------------%
[parentdir,~,~]=fileparts(pwd);
global validdata
[validdata] = textread(strcat(parentdir,'/Data/digitsvalid.txt'),'','delimiter',',');
nVSamples = size(validdata,1);

%----preprocess----%
data_mean = mean(mean(traindata(:,1:end-1)));
data_std = std(std(traindata(:,1:end-1)));
traindata(:,1:end-1) = (traindata(:,1:end-1)); %-data_mean)/data_std;
validdata(:,1:end-1) = (validdata(:,1:end-1)); %-data_mean)/data_std;
%---shuffle the data-----%
traindata = traindata(randperm(size(traindata,1)),:);

%---Model Definition-----%
AE_arr = [784,100,784]
global model
lr = 0.01;
epochs = 100;
batchsize = 1;
dropout_val = 0.5;

nHiddens = [50, 100, 200. 500];
train_NLL_Err = zeros(epochs,4);
valid_NLL_Err = zeros(epochs,4);

for nH =  1:size(nHiddens,2)
    AE_arr = [784,nHiddens(nH),784]
    model = AutoEncoder.define_model(AE_arr,dropout_val);
    disp(model)
    
    for i = 1:epochs
        for j = nSamples:-1:1
            data = AutoEncoder.mySignum(traindata(j,1:end-1)');
            model = AutoEncoder.fprop(data,model,1);
            if i>1
                model = AutoEncoder.bprop(model);
                model = AutoEncoder.updateParams(model,lr);
            end
        end
        for j = nSamples:-1:1
            data = AutoEncoder.mySignum(traindata(j,1:end-1)');
            model = AutoEncoder.fprop(data,model,0);
            train_NLL_Err(i,nH) = train_NLL_Err(i,nH)+AutoEncoder.myCrossEntropy(model);
        end
        train_NLL_Err(i,nH) = train_NLL_Err(i,nH)/nSamples;
        for j = nVSamples:-1:1
            data = AutoEncoder.mySignum(validdata(j,1:end-1)');
            model = AutoEncoder.fprop(data,model,0);
            valid_NLL_Err(i,nH) = valid_NLL_Err(i,nH)+AutoEncoder.myCrossEntropy(model);
        end
        valid_NLL_Err(i,nH) = valid_NLL_Err(i,nH)/nVSamples;
    end
end
k=2
figure, 
subplot(1,2,1),plot(train_NLL_Err(k:end,1)), hold on
subplot(1,2,1),plot(train_NLL_Err(k:end,2)), hold on 
subplot(1,2,1),plot(train_NLL_Err(k:end,3)), hold on
subplot(1,2,1),plot(train_NLL_Err(k:end,4)) 
legend('DAE 50','DAE 100','DAE 200', 'DAE 500', 'Location','northwest')
title('DAE - Training Cross Entropy Error')
xlabel('Epochs')
ylabel('Error')

subplot(1,2,2),plot(valid_NLL_Err(k:end,1)), hold on
subplot(1,2,2),plot(valid_NLL_Err(k:end,2)), hold on
subplot(1,2,2),plot(valid_NLL_Err(k:end,3)), hold on
subplot(1,2,2),plot(valid_NLL_Err(k:end,4)) 
legend('DAE 50','DAE 100','DAE 200', 'DAE 500', 'Location','northwest')
title('DAE - Validation Cross Entropy Error')
xlabel('Epochs')
ylabel('Error')
toc/60