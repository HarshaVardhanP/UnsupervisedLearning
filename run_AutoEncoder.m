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
model = AutoEncoder.define_model(AE_arr,0);
disp(model)

train_NLL_Err = zeros(epochs,1);
valid_NLL_Err = zeros(epochs,1);
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
        train_NLL_Err(i) = train_NLL_Err(i)+AutoEncoder.myCrossEntropy(model);
    end
    train_NLL_Err(i) = train_NLL_Err(i)/nSamples;
    train_NLL_Err(i)
    for j = nVSamples:-1:1
        data = AutoEncoder.mySignum(validdata(j,1:end-1)');
        model = AutoEncoder.fprop(data,model,0);
        valid_NLL_Err(i) = valid_NLL_Err(i)+AutoEncoder.myCrossEntropy(model);
    end
    valid_NLL_Err(i) = valid_NLL_Err(i)/nVSamples;
    valid_NLL_Err(i)
end
plot(train_NLL_Err(2:end)), hold on
plot(valid_NLL_Err(2:end))
legend('Training','Validation')
save('model_autoencoder.mat')
visualizeImgs(model.weights{1})
