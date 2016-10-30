classdef NN
    % Modules of Neural Network
    % Includes various activation functions    
    properties
    end
    
    methods (Static)
        % Random Binary Sequence with 0.5 probability 
        function x = randBinary(n,val)
            % Total length 'n' must be even to have even # of 0s and 1s.
            nOnes = n*val;
            % List of random locations.
            indexes = randperm(n);
            x = ones(n,1);
            if nOnes == 0
                % do nothing
            else
                x(indexes(1:nOnes)) = 0;
            end
        end
        
        %----------Define Network Architecture--------%
        %----inputs : layer_arr = [nInputs,nHidden_1,...nHidden_i,...nOutputs]---%
        %-------------Example = [784,100,10]--------%
        %-------------Linear,Sigmoid, Linear, Sigmoid, Softmax ----------- %
        function model = define_model(layer_arr,dropout,batchsize)
            model.data={};
            model.dropout_val = dropout;

            model.weights = {};
            model.gradweights = {};
            model.prevgradweights = {}; %--for momentum

            model.biases = {};
            model.gradbiases = {};
            model.prevgradbiases = {}; %---for momentum

            model.preacts = {};
            model.gradpreacts = {};
            model.dropoutvector = {};

            model.hiddens = {};
            model.gradhiddens = {};

            for i = 1:size(layer_arr,2)-1
               init_val = sqrt(6)/sqrt(layer_arr(i)+layer_arr(i+1));
               init_val = 0.01;
               %-----Params----%
               model.weights{i} = (rand(layer_arr(i),layer_arr(i+1))-0.5)*2*init_val;  % Uniform random distribution in [-0.1,0.1]
               model.biases{i} = (rand(layer_arr(i+1),1)-0.5)*2*init_val;
               model.preacts{i} = zeros(layer_arr(i+1),1);

               %-----Gradients-----%
               model.gradweights{i} = zeros(layer_arr(i),layer_arr(i+1));
               model.gradbiases{i} = zeros(layer_arr(i+1),1);
               model.prevgradweights{i} = zeros(layer_arr(i),layer_arr(i+1));
               model.prevgradbiases{i} = zeros(layer_arr(i+1),1);

               model.gradpreacts{i} = zeros(layer_arr(i+1),1);
            end
            for i = 1:size(layer_arr,2)-2
                model.hiddens{i} = zeros(layer_arr(i+1),1);
                model.gradhiddens{i} = zeros(layer_arr(i+1),1);
                if model.dropout_val > 0
                    model.dropoutvector{i} = NN.randBinary(layer_arr(i+1),model.dropout_val);
                else
                    model.dropoutvector{i} = ones(layer_arr(i+1),1);
                end
            end
        end
        
        %--------------F-Prop-------------------%
        %----phase = 1 for training and phase = 0 for testing----% 
        function [Y,model] = fprop(data,model,phase)
            nLayers = size(model.weights,2);
            model.data = data(:,1:end-1)';
            A = model.data;
            for i = 1:nLayers
               A = [A;1];
               size(A);
               % Linear %
               W = [model.weights{i} ; model.biases{i}'];
               A = W'*A;
               model.preacts{i} = A;
               if i < nLayers
                % Sigmoid Activation %
                A = NN.mySigmoid(A);
                if phase == 1
                    model.dropoutvector{i} = NN.randBinary(size(model.dropoutvector{i},1),model.dropout_val); 
                    A = A.*model.dropoutvector{i};
                else
                    A = A.*ones(size(model.dropoutvector{i},1),1)*0.5; 
                end
                model.hiddens{i} = A;
               end
            end
            % Softmax 
            Y = NN.mySoftmax(A);
        end
        
        
        %------Backward Propagation------%
        function model = bprop(LossGrad,model,y,t)
            nLayers = size(model.weights,2);
            %-----Output Gradient----%  
            T = zeros(10,1);
            T(t+1)= 1;
            Op = -(T-y);
            Op;
            model.gradpreacts{nLayers} = Op;
            for nL = nLayers:-1:1
                model.prevgradweights{nL} = model.gradweights{nL};
                model.prevgradbiases{nL} = model.gradbiases{nL};
                if nL > 1
                    %-- Weights & Biases --%
                    model.gradweights{nL} =    model.hiddens{nL-1} * model.gradpreacts{nL}' ; 
                    model.gradbiases{nL} = model.gradpreacts{nL}; 
                    %--
                    model.gradhiddens{nL-1} =  model.weights{nL} * model.gradpreacts{nL} ;
                    model.gradhiddens{nL-1} =  model.gradhiddens{nL-1}.*model.dropoutvector{nL-1};
                    %size(model.gradhiddens{nL-1})
                    %---For Sigmoid--%
                    a = model.preacts{nL-1};
                    gradActs = NN.mySigmoid(a).*(1-NN.mySigmoid(a));
                    %size(gradActs)
                    model.gradpreacts{nL-1} = model.gradhiddens{nL-1}.* gradActs;
                end
                if nL == 1  %---First Layer - No updates to inputs/activations
                    %-- Weights & Biases --%
                    model.gradweights{nL} =  (model.data) * model.gradpreacts{nL}'; 
                    model.gradbiases{nL} = (model.gradpreacts{nL}); 
                end
            end 
        end
        
        
        %--------Update Parameters with Gradients---%
        function model = updateParams(model,lr,mu)
             nLayers = size(model.weights,2);
             for i=1:nLayers
                 model.weights{i} = model.weights{i}-lr*(model.gradweights{i})-lr*mu*(model.prevgradweights{i}); %(0.001*model.weights{i});
                 model.biases{i} = model.biases{i}-lr*(model.gradbiases{i})-lr*mu*(model.prevgradbiases{i});         
             end
        end
        
        
        %------Validation Data - FPROP------%
        function [NLLerr, Cerr, OPs] = run_valid(validdata,model)
            nVSamples = size(validdata,1);
            res = -1*ones(nVSamples,1);
            acc = 0;
            NLLerr = 0;
            OPs = [];
            test_phase = 0;
            for j = 1:nVSamples
               [Y,model] = NN.fprop(validdata(j,:),model,test_phase);
               OPs = [OPs Y];
               target = validdata(j,end);
               [val,idx] = max(Y);
               res(j) = idx-1;
               if res(j) == target
                   acc = acc+1;
                   %disp('Hi')
                end
                %----Loss Function : Cross Entropy Error----%
                [Error,LossGrad] = NN.myCrossEntropy(Y,target);
                NLLerr = NLLerr+Error;      
            end
            Cerr = (acc)/nVSamples;
            Cerr = Cerr * 100;
            NLLerr = NLLerr/nVSamples;
        end
       
        
        % Sigmoid Activation
        function y = mySigmoid(x)
            y = 1./(1+exp(-x));
        end
        
        % Softmax Activation
        % Arg : x is input matrix ; nSamples x nClasses
        function y = mySoftmax(x)
            x_sum = sum(exp(x'))';
            x_sum_repmat = repmat(x_sum,1,size(x,2));
            y = exp(x)./x_sum_repmat;
        end
        
        % Cross-Entropy-Error
        % Arg:  y is output of fprop ; Dim : nSamples x nClasses
        %       t is Target Class Labels; Dim : nSamples x 1
        function [err,grad] = myCrossEntropy(y,t)
            % -- Loss Error Calculation--%
            err = -log(y(t+1));                                                     
            %----Loss Gradient Calculation --- %
            grad = -1/(y(t+1));                            
        end        
    end
end

