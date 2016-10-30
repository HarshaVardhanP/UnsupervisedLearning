tic
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
RBM_arr = [784,100]
global model
lr = 0.01;
GS_steps = 1;
epochs = 100;
batchsize = 1;
model = RBM.define_model(RBM_arr);
disp(model)

%----Training------%
train_phase = 1;
train_NLL_Err = zeros(epochs,1);
valid_NLL_Err = zeros(epochs,1);
for i = 1:epochs
    for j = nSamples:-1:1
        %----create Negative Sample using K steps Gibbs Sampling ------%
        model.X{1} = RBM.mySignum(traindata(j,1:end-1)');
        model = RBM.ContrastiveDivergence(model,GS_steps);
        %train_NLL_Err(i) = train_NLL_Err(i)+RBM.myCrossEntropy(model);
        %----Update Parameters-----%
        if i > 1
            model = RBM.updateParams(model,lr);
        end
    end
    train_NLL_Err(i) = RBM.calc_error(model,traindata,GS_steps);
    valid_NLL_Err(i) = RBM.calc_error(model,validdata,1);
end
plot(train_NLL_Err(2:end)), hold on
plot(valid_NLL_Err(2:end))
legend('Training','Validation')
title('Cross Entropy Error')
xlabel('Epochs')
ylabel('Error')
%save('model_5a_best')
visualizeImgs(model.weights{1})
%toc/60