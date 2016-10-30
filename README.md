# Unsupervised Learning (RBM, Auto-Encoder, Denoising Auto-Encoder)

Objective : To implement unsupervised neural networks and verify which pretrained filters are better for Digit classification among RBM, AutoEncoder and Denoisy-AutoEncoder. 

Dataset: MNIST Subset<br />
Training Set : 3000 Samples [300 x 10]<br />
Validation Set : 1000 Samples [100 x 10]<br />
Testing Set : 3000 [300 x 10]<br />

## Neural Network Architecture:
Inputs [784] -> FCN1[100] -> Sigmoid -> Dropout[0.5] -> Outputs[10]

## Restricted Boltzmann Machines:

Architecture : Visible Units [784] -> FCN1[100] -> Sigmoid -> Hidden units [100]  & Cross Entropy Loss

Cross Entropy Error plot on Training and Validation Set<br />
![5a_e](https://cloud.githubusercontent.com/assets/5204400/19833674/aa502ad2-9e18-11e6-8c8e-27d0754244d7.jpg)

RBM Weight Filters Visulaization<br />
![5a_f](https://cloud.githubusercontent.com/assets/5204400/19833676/aa51d760-9e18-11e6-9a26-02ba0f0ead20.jpg)

Performance comparison on NN using Random Weights Initialization and RBM Weights Initialization
![5d_c](https://cloud.githubusercontent.com/assets/5204400/19833673/aa4f6f2a-9e18-11e6-8e0b-6f70c43252c2.jpg)

## Auto-Encoders:
Architecture : Inputs [784] -> FCN1[100]-> Sigmoid -> Hidden units [100] -> FCN1[784] -> Sigmoid -> Outputs [784]
Loss : Cross Entropy Loss & Weights are tied

Performance comparison on NN using RBM Weights Initialization and AutoEncoder Weights Initialization
![5e_c](https://cloud.githubusercontent.com/assets/5204400/19833675/aa501c22-9e18-11e6-83ca-3da8c07488e1.jpg)

AutoEncoder Weight Filters Visulaization<br />
![5e_f](https://cloud.githubusercontent.com/assets/5204400/19833672/aa4dbdec-9e18-11e6-9a80-14618e21dafe.jpg)

## Denoising Auto-Encoders:
Architecture : Inputs [784] -> Dropout -> mInputs[784] -> FCN1[100]-> Sigmoid -> Hidden units [100] -> FCN1[784] -> Sigmoid -> Outputs [784]
Loss : Cross Entropy Loss & Weights are tied

Performance comparison on NN using AutoEncoder Weights Initialization and Denoisy AutoEncoder Weights Initialization
![5f_c](https://cloud.githubusercontent.com/assets/5204400/19833671/aa4d1568-9e18-11e6-8f18-e5cddc96c599.jpg)

Denoising AutoEncoder Weight Filters Visulaization<br />
![5f_f](https://cloud.githubusercontent.com/assets/5204400/19833678/aa533a10-9e18-11e6-9916-8ae42fe38e89.jpg)

Training RBMs, autoencoders and denoising autoencoders with 50, 100, 200, and 500 hidden units to observe the
effect of this modification on the convergence properties, and the generalization of the network.
![5g_ae](https://cloud.githubusercontent.com/assets/5204400/19833679/aa566514-9e18-11e6-8c43-a2294fed8310.jpg)
![5g_dae](https://cloud.githubusercontent.com/assets/5204400/19833680/aa568f4e-9e18-11e6-999e-15479711e7ce.jpg)
![5g_rbm](https://cloud.githubusercontent.com/assets/5204400/19833681/aa59ad82-9e18-11e6-9037-33c7346a30ef.jpg)
