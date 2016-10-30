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
NN_arr  = [784,100,10];
dropout = 0.5;
lr = 0.1;
mu = 0;
epochs = 50;
batchsize = 1; 
global model;

%-------------------------------%

train_NLLerr = zeros(epochs,2);
train_Cerr = zeros(epochs,2);
valid_NLLerr = zeros(epochs,2);
valid_Cerr = zeros(epochs,2);


for init = 1:2
    model = NN.define_model(NN_arr,dropout,batchsize);
    disp(model)
    if i==2
        %---Model Initialization with Weights----%
        RBM = load('Models/model_5a_best.mat');
        RBM_model = RBM.model;
        RBM_W = RBM_model.weights{1};
        model.weights{1}=RBM_W;
    end
    %---------Train the model----------%
    % phase = 1 for Training, phase = 0 for Testing;
    train_phase = 1; % always 1
    test_phase = 0; % always 0
    for i=1:epochs
        res = -1*ones(nSamples,1);
        Ops = [];
        for j = nSamples:-1:1
            %----Forward Prop-------%
            [Y,model] = NN.fprop(traindata(j,:),model,train_phase);
            Ops = [Ops Y];
            target = traindata(j,end);
            [val,idx] = max(Y);
            res(j) = idx-1;
            if res(j) == target
               train_Cerr(i,init) = train_Cerr(i,init)+1;
            end
            %----Loss Function : Cross Entropy Error----%
            [Error,LossGrad] = NN.myCrossEntropy(Y,target);
            train_NLLerr(i,init) = train_NLLerr(i,init)+Error;
            if i > 0
                %----Backward Prop------%
                model = NN.bprop(LossGrad,model,Y,target);
                %----Update Weights and Biases-----%
                model = NN.updateParams(model,lr,mu);
            end
        end    
        train_NLLerr(i,init) = train_NLLerr(i,init)/nSamples;
        train_Cerr(i,init) = (train_Cerr(i,init)/nSamples)*100;
        %disp(acc)
        %------------VALIDATION---------------%
        [valid_NLLerr(i,init), valid_Cerr(i,init),OPs] = NN.run_valid(validdata,model); 
    end
end
figure, 
subplot(1,2,1),plot(train_Cerr(:,1)), hold on
subplot(1,2,1),plot(train_Cerr(:,2)) 
legend('Train - Random','Train - RBM Pretrained','Location','northwest')
title('Training Classification Accuracy')

subplot(1,2,2),plot(valid_Cerr(:,1)), hold on
subplot(1,2,2),plot(valid_Cerr(:,2)) 
legend('Valid - Random','Valid - RBM Pretrained','Location','northwest')
title('Validation Classification Accuracy')


%figure, plot(res)
