classdef AutoEncoder
    % Modules of Neural Network
    % Includes various activation functions    
    properties
    end
    methods (Static)
        %----------Define Network Architecture--------%
        %----inputs : layer_arr = [nInputs,nHiddens,nOutputs], Tied Weights---%
        %-------------Example = [784,100,784]-------- %
        %-------------Linear, Sigmoid, Linear Sigmoid----------- %
        function model = define_model(layer_arr,dropout)
            model.inputs={};
            model.dropoutvector={};
            model.dropout_val = dropout;
            
            model.weights = {};
            model.gradweights = {};
           
            model.biases = {};
            model.gradbiases = {};
            
            model.preacts = {};
            model.gradpreacts = {};
            
            model.hiddens = {};
            model.gradhiddens = {};
            
            model.outputs = {};
            
            model.inputs{1} =  zeros(layer_arr(1),1);
            if model.dropout_val > 0
                model.dropoutvector{1} = AutoEncoder.randBinary(layer_arr(1),model.dropout_val);
            else
                model.dropoutvector{1} = ones(layer_arr(1),1);
            end
            %init_val = sqrt(6)/sqrt(layer_arr(i)+layer_arr(i+1));
            init_val = 0.01;
            for i = 1:size(layer_arr,2)-1
               %-----Params----%
               model.biases{i} = (rand(layer_arr(i+1),1)-0.5)*2*init_val;
               model.preacts{i} = zeros(layer_arr(i+1),1);
                
               %-----Gradients-----%
               model.gradbiases{i} = zeros(layer_arr(i+1),1);
               model.gradpreacts{i} = zeros(layer_arr(i+1),1);
               model.gradweights{i} = zeros(layer_arr(i),layer_arr(i+1));
            end
            for i = 1:size(layer_arr,2)-2
                model.weights{i} = (rand(layer_arr(i),layer_arr(i+1))-0.5)*2*init_val;  % Uniform random distribution in [-0.1,0.1]
                model.hiddens{i} = zeros(layer_arr(i+1),1);
                model.gradhiddens{i} = zeros(layer_arr(i+1),1);
            end
            model.outputs{1} =  zeros(layer_arr(end),1);
        end
        
        %--------------F-Prop-------------------% 
        %----Considering model for 784,100,784---%
        function model = fprop(data,model,phase)
            nLayers = size(model.weights,2);
            model.inputs{1} = data;
            if phase == 1
                model.dropoutvector{1} = AutoEncoder.randBinary(size(model.dropoutvector{1},1),model.dropout_val); 
                model.inputs{1} = model.inputs{1}.*model.dropoutvector{1};
            else
                if model.dropout_val == 0
                    val = 0;
                else
                    val = model.dropout_val;
                end
                model.inputs{1} = model.inputs{1}.*ones(size(model.dropoutvector{1},1),1); 
            end
            model.preacts{1} = model.biases{1}+model.weights{1}'*model.inputs{1};
            model.hiddens{1} = AutoEncoder.mySigmoid(model.preacts{1});
            %model.hiddens{1} = AutoEncoder.mySignum(model.hiddens{1});
            model.preacts{2} = model.biases{2}+model.weights{1}*model.hiddens{1};
            model.outputs{1} = AutoEncoder.mySigmoid(model.preacts{2});
            %model.outputs{1} = AutoEncoder.mySignum(model.outputs{1});
        end
   
        %------Backward Propagation------%
        function model = bprop(model)
            %nLayers = size(model.weights,2);
            %-----Output Gradient----%  
            %T = zeros(10,1);
            %T(t+1)= 1;
            %Op = -(T-y);
            Op = -(model.inputs{1}-model.outputs{1});
            Op;
         
            model.gradpreacts{2} = Op; %.* gradActs;
            
            % weights & biases
            model.gradweights{2} = model.hiddens{1} * model.gradpreacts{2}' ; 
            model.gradbiases{2} = model.gradpreacts{2};
            model.gradhiddens{1} =  model.weights{1}' * model.gradpreacts{2} ;
            % 
            a = model.preacts{1};
            gradActs = AutoEncoder.mySigmoid(a).*(1-AutoEncoder.mySigmoid(a));
            model.gradpreacts{1} = model.gradhiddens{1}.* gradActs;
            %
            model.gradweights{1} =  (model.inputs{1}) * model.gradpreacts{1}'; 
            model.gradbiases{1} = (model.gradpreacts{1}); 
        end
        
        %----- Calculate gradients and update the parameters
        function model = updateParams(model,lr)
           %----- Update Gradients ------%
            model.biases{2} = model.biases{2}-lr*model.gradbiases{2};
            model.weights{1} = model.weights{1}-lr*(model.gradweights{1}+model.gradweights{2}');
            model.biases{1} = model.biases{1}-lr*model.gradbiases{1};
        end 
        
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
        
        
        % Sigmoid Activation
        function y = mySigmoid(x)
            y = 1./(1+exp(-x));
        end
        
        % Signum Function (<0.5 = 0 & >0.5 = 1) 
        function y = mySignum(x) 
            y = x;
            y(y<0.5)=0;
            y(y>=0.5)=1;
        end
                      
        % Cross-Entropy-Error
        %  xi Log(Sigmoid(Wih+ci)) 
        % Arg:  y is output of fprop ; Dim : nSamples x nClasses
        function err = myCrossEntropy(model)
            x = model.inputs{1};
            t = model.outputs{1};
            err = -1*sum(x.*log(t)+(1-x).*log(1-t));     
            %err = err/size(x,1);
        end        
        
    end
end

