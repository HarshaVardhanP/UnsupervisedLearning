# Unsupervised Learning (RBM, Auto-Encoder, Denoising Auto-Encoder)

Objective : To implement unsupervised neural networks and verify which pretrained filters are better for Digit classification among RBM, AutoEncoder and Denoisy-AutoEncoder. 

Dataset: MNIST Subset<br />  
Training Set : 3000 Samples [300 x 10]<br />
Validation Set : 1000 Samples [100 x 10]<br />
Testing Set : 3000 [300 x 10]<br />

Restricted Boltzmann Machines:

Architecture : Visible Units [784] -> Hidden units [100]
Loss used: Cross Entropy Loss

Cross Entropy Error plot on Training and Validation Set<br />
![5a_e](https://cloud.githubusercontent.com/assets/5204400/19833674/aa502ad2-9e18-11e6-8c8e-27d0754244d7.jpg)

Weight Filters Visulaization<br />
![5a_f](https://cloud.githubusercontent.com/assets/5204400/19833676/aa51d760-9e18-11e6-9a26-02ba0f0ead20.jpg)
