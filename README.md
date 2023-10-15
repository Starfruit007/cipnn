# **CIPNN - Continuous Indeterminate Probability Neural Network**  

Pytorch implementation of paper:  **Continuous Indeterminate Probability Neural Network**,  


## **Environment**

1. Our environment is: Python 3.10.11  torch==2.0.1, torchvision==0.15.2  
    > pip install -r requirements.txt  

## **Quick Start**

1. Run CIPNN on Datasets to check classfication performance.  
    > python3 demo_mnist.py --num_epoch 10 --gamma 0.9 --num_latent_vars 3 --learning_rate 1e-3   --forget_num 3000  
    > python3 demo_fashionmnist.py --num_epoch 10 --gamma 0.9 --num_latent_vars 3 --learning_rate 1e-3   --forget_num 3000  
    > python3 demo_cifar10.py --num_epoch 10 --gamma 0.85 --num_latent_vars 3 --learning_rate 1e-4   --forget_num 3000   
    > python3 demo_stl10.py --num_epoch 10 --gamma 0.9 --num_latent_vars 3 --learning_rate 1e-4   --forget_num 3000   

2. Run CIPNN on large latent space.   
    > python3 demo_mnist.py --num_epoch 10 --gamma 0.9 --learning_rate 1e-3 --forget_num 500 --num_latent_vars 1000 --scaler_factor 4.05  
    > python3 demo_mnist.py --num_epoch 10 --gamma 0.9 --learning_rate 1e-3 --forget_num 500 --num_latent_vars 500   
    > python3 demo_mnist.py --num_epoch 10 --gamma 0.9 --learning_rate 1e-3 --forget_num 500 --num_latent_vars 200  
    > python3 demo_mnist.py --num_epoch 10 --gamma 0.9 --learning_rate 1e-3 --forget_num 500 --num_latent_vars 100   
    > python3 demo_mnist.py --num_epoch 10 --gamma 0.9 --learning_rate 1e-3 --forget_num 500 --num_latent_vars 50  
    > python3 demo_mnist.py --num_epoch 10 --gamma 0.9 --learning_rate 1e-3 --forget_num 500 --num_latent_vars 20  
    > python3 demo_mnist.py --num_epoch 10 --gamma 0.9 --learning_rate 1e-3 --forget_num 500 --num_latent_vars 10  
    > python3 demo_mnist.py --num_epoch 10 --gamma 0.9 --learning_rate 1e-3 --forget_num 500 --num_latent_vars 5    

3. Run CIPNN on MNIST to check the auto-encoder task performance with different latent spaces.  
    > python3 demo_autoencoder_mnist.py --mnist_data_path ../ --num_epoch 10 --gamma 0.99 --num_latent_vars 1  
    > python3 demo_autoencoder_mnist.py --mnist_data_path ../ --num_epoch 10 --gamma 0.98 --num_latent_vars 2  
    > python3 demo_autoencoder_mnist.py --mnist_data_path ../ --num_epoch 10 --gamma 0.93 --num_latent_vars 10    


## **Quick Results Check**

The logs of above commands are stored into log folder, you can easily access them if you do want.



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Starfruit007/cipnn&type=Date)](https://star-history.com/#Starfruit007/cipnn&Date)