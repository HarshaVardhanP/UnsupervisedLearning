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

nHiddens = [50, 100, 200. 500];
train_NLL_Err = zeros(epochs,4);
valid_NLL_Err = zeros(epochs,4);

for nH = 1:size(nHiddens,2)
    RBM_arr = [784, nHiddens(nH)]
    model = RBM.define_model(RBM_arr);
    disp(model)

    %----Training------%
    train_phase = 1;
   
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
        train_NLL_Err(i,nH) = RBM.calc_error(model,traindata,GS_steps);
        valid_NLL_Err(i,nH) = RBM.calc_error(model,validdata,1);
    end
end

k=2
figure, 
subplot(1,2,1),plot(train_NLL_Err(k:end,1)), hold on
subplot(1,2,1),plot(train_NLL_Err(k:end,2)), hold on 
subplot(1,2,1),plot(train_NLL_Err(k:end,3)), hold on
subplot(1,2,1),plot(train_NLL_Err(k:end,4)) 
legend('RBM 50','RBM 100','RBM 200', 'RBM 500', 'Location','northwest')
title('RBM - Training Cross Entropy Error')
xlabel('Epochs')
ylabel('Error')

subplot(1,2,2),plot(valid_NLL_Err(k:end,1)), hold on
subplot(1,2,2),plot(valid_NLL_Err(k:end,2)), hold on
subplot(1,2,2),plot(valid_NLL_Err(k:end,3)), hold on
subplot(1,2,2),plot(valid_NLL_Err(k:end,4)) 
legend('RBM 50','RBM 100','RBM 200', 'RBM 500', 'Location','northwest')
title('RBM - Validation Cross Entropy Error')
xlabel('Epochs')
ylabel('Error')

%save('model_5a_best')
%visualizeImgs(model.weights{1})
toc/60