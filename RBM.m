classdef RBM
    % Modules of Neural Network
    % Includes various activation functions    
    properties
    end
    methods (Static)
        %----------Define Network Architecture--------%
        %----inputs : layer_arr = [nInputs,nHiddens]---%
        %-------------Example = [784,100]--------%
        %-------------Linear, Sigmoid----------- %
        function model = define_model(layer_arr)
            model.X={};
            model.Xneg = {};
            model.weights = {};
            model.gradweights = {};
            model.biases_b = {};
            model.gradbiases_b = {};
            model.biases_c = {};
            model.gradbiases_c = {};
            model.hiddens_x = {};
            model.hiddens = {};
            i = 1;
            %init_val = sqrt(6)/sqrt(layer_arr(i)+layer_arr(i+1));
            init_val = 0.1;
            %-----Params----%
            model.X{i} = zeros(layer_arr(i),1);
            model.Xneg{i}= zeros(layer_arr(i),1);
            model.weights{i} = (rand(layer_arr(i),layer_arr(i+1))-0.5)*2*init_val;  % Uniform random distribution in [-0.1,0.1]
            model.biases_b{i} = zeros(layer_arr(i+1),1); %(rand(layer_arr(i+1),1)-0.5)*2*init_val;
            model.biases_c{i} = zeros(layer_arr(i),1); %(rand(layer_arr(i),1)-0.5)*2*init_val;
            model.hiddens_x{i} = zeros(layer_arr(i+1),1); %RBM.mySignum(rand(layer_arr(i+1),1));  
            model.hiddens{i} = zeros(layer_arr(i+1),1); 
            
            %-----Gradients-----%
           model.gradweights{i} = zeros(layer_arr(i),layer_arr(i+1));
           model.gradbiases_b{i} = zeros(layer_arr(i+1),1);
           model.gradbiases_c{i} = zeros(layer_arr(i),1);
        end
        
        
        %---Contrastive Divergence
        function model = ContrastiveDivergence(model,K) 
            % Initialize Xo = Xpos
            % Sample ho from P(h|Xo) ; p = sigmoid(bj+WjX) if p > 0.5 hj = 1  
            % For t  = 1:K
            %   - Sample x_t from P(X|h_t-1)
            %   - Sample h_t from P(h|x_t)
            x = model.X{1};
            W = model.weights{1};
            b = model.biases_b{1};
            c = model.biases_c{1};
            phx =  RBM.mySigmoid(b+W'*x);
            model.hiddens_x{1} = RBM.GibbsSample(model.hiddens{1},phx);
            model.hiddens{1} = model.hiddens_x{1};
            for t = 1:K
                pxh = RBM.mySigmoid(c+W*model.hiddens{1});
                model.Xneg{1} = RBM.GibbsSample(model.Xneg{1},pxh);
                phx = RBM.mySigmoid(b+W'*model.Xneg{1});
                model.hiddens{1} = RBM.GibbsSample(model.hiddens{1},phx);
            end
        end
        
        
        % Gibbs Sample
        function y = GibbsSample(x,p)
            y = rand(size(x));
%              y(y<p)=0;
%              y(y>p)=1;
            ca = find(y>p);
            cb = find(y<=p);
            y(ca) =0;
            y(cb) =1;
        end
        
        
        %----- Calculate gradients and update the parameters
        function model = updateParams(model,lr)
            %  W = W + lr*(H(Xpos).Xpos - H(Xneg).Xneg)
            %  b = b + lr*(H(Xpos)-H(Xneg))
            %  c = c + lr*(Xpos-Xneg)

            %---- 1. Calculation of Gradients ------%
            x_pos = model.X{1};
            x_neg = model.Xneg{1};
            W = model.weights{1};
            b = model.biases_b{1};

            h_pos = model.hiddens_x{1}; %(RBM.mySigmoid(b+W'*x_pos));
            h_neg = model.hiddens{1}; %(RBM.mySigmoid(b+W'*x_neg));

            model.gradweights{1} = x_pos*h_pos'-x_neg*h_neg';
            model.gradbiases_b{1} = h_pos-h_neg;
            model.gradbiases_c{1} = x_pos-x_neg;

            %----- 2. Update Gradients ------%
            model.weights{1} = model.weights{1}+lr*model.gradweights{1};
            model.biases_b{1} = model.biases_b{1}+lr*model.gradbiases_b{1};
            model.biases_c{1} = model.biases_c{1}+lr*model.gradbiases_c{1};
        end
        
        %----Calculate Error on ValidData
        function err = calc_error(model,validdata,K)
            err = 0;
            nVSamples = size(validdata,1);
            for j = 1:nVSamples
                %----create Negative Sample using K steps Gibbs Sampling ------%
                model.X{1} = RBM.mySignum(validdata(j,1:end-1)');
                model = RBM.ContrastiveDivergence(model,K);
                err = err+RBM.myCrossEntropy(model);
            end
            err = err/nVSamples;
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
            x = model.X{1};
            t = RBM.mySigmoid(model.weights{1}*model.hiddens{1}+model.biases_c{1});
            err = -1*sum(x.*log(t)+(1-x).*log(1-t));     
            %err = err/size(x,1);
        end        
        
    end
end

