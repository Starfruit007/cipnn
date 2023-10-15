import torch
from torchvision import datasets
from torchvision import transforms
from src.cipnn import CIPNN
from src.trainer import Exp_Trainer
import matplotlib.pyplot as plt
import numpy as np
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


class MNIST_CIPNN(torch.nn.Module):
    def __init__(self,args):
        super().__init__()

        assert args.num_params == 2 and  \
            (len(args.gamma) == 1 or len(args.gamma) == args.num_latent_vars),\
                'parameter setting error.'

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, args.num_params*args.num_latent_vars),

        )

        # self.classifier.apply(self.init_weights) # used in IPNN in order to avoid local minimum at training begining.
        args_dict = dict(model_type = 'cipnn',
                        forget_num = args.forget_num, 
                        stable_num = args.stable_num, 
                        monte_carlo_num = args.monte_carlo_num,
                        gamma = torch.tensor(args.gamma).to(device),
                        beta = args.beta,
                        num_params = args.num_params,
                        scaler_factor = args.scaler_factor)

        self.cipnn = CIPNN(**args_dict) # for classification
        args_dict['model_type'] = 'cipae'
        self.cipae = CIPNN(**args_dict) # in order to reconstruct the latent variable to see what they have learned
        

    def forward(self, images, labels = None):
        
        images = torch.reshape(images,[images.shape[0],-1]).to(device)    
        if labels is not None: 
            labels = labels.to(device)
            y_true = torch.nn.functional.one_hot(labels,10).float().to(device)
            y_true_ae = images
        else: 
            y_true = None
            y_true_ae = None    
            
        logits = self.classifier(images)

        cls_outputs = self.cipnn(logits,[y_true]) # related information will be stored into cipnn.recorder.

        ae_outputs = self.cipae(logits,[y_true_ae]) # related information will be stored into cipae.recorder.

        loss = sum(cls_outputs['losses'])
        outputs = cls_outputs

        return loss,outputs
        
    def init_weights(self,m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.uniform_(m.weight,a=-0.3, b=0.3)
            torch.nn.init.uniform_(m.bias,a=-0.3, b=0.3)

def parse_args():
    parser = argparse.ArgumentParser(description="Set Parameters for IPNN - Indeterminate Probability Neural Network.")
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=5,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3, 
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0, 
        help="weight decay",
    )
    parser.add_argument(
        "--num_latent_vars",
        type=int,
        default=2,
        help="dimension of latent variables",
    )
    parser.add_argument(
        "--num_params",
        type=int,
        default=2,
        help="number of prior distributions' parameter, for Gaussian distribution: 2 (mean and variance)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1, 
        help="beta value to trade off the balance between main loss and KL loss.",
    )
    parser.add_argument(
        "--gamma",
        nargs='+', 
        type=float,
        default=[0.9], 
        help="gamma value to regularize the conditional joint distributions.",
    )
    parser.add_argument(
        "--monte_carlo_num",
        type=float,
        default=2, 
        help="Monte Carlo number C",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='../', 
        help="data path",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64, 
        help="train batch size",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128, 
        help="eval batch size",
    )
    parser.add_argument(
        "--forget_num",
        type=int,
        default=3000, 
        help="forget number T",
    )
    parser.add_argument(
        "--stable_num",
        type=float,
        default=1e-20, 
        help="stable number (or epsilon in IPNN) for training",
    )
    parser.add_argument(
        "--scaler_factor",
        type=float,
        default=4.13273, 
        help="expectation scaler factor for large latent space.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=' ', 
        help="results saved file path",
    )
    args = parser.parse_args()

    return args

def main(i,sd,log):
    args = parse_args()
    if i == 0: log.add_logger_info(args)
    # Download the MNIST Dataset
    tensor_transform = transforms.ToTensor() # Transforms images to a PyTorch Tensor
    train_dataset = datasets.MNIST(root = args.data_path,train = True,download = False,transform = tensor_transform)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.train_batch_size,num_workers = 0, shuffle = True)
    eval_dataset = datasets.MNIST(root = args.data_path,train = False,download = False,transform = tensor_transform)
    eval_loader = torch.utils.data.DataLoader(dataset = eval_dataset,batch_size = args.eval_batch_size,num_workers = 0, shuffle = False)

    # Model Initialization
    model = MNIST_CIPNN(args)
    model.to(device)

    exp = Exp_Trainer(model,train_loader,eval_loader,args.num_epoch,args.learning_rate, args.weight_decay)
    recorder_dict = exp.exp_start()

    recorder_dict['args'] = args
    sd.main(recorder_dict,args.save_path)

    return recorder_dict
    

if __name__ == "__main__":

    from src.logger import CreateLogger,SaveData

    log = CreateLogger('logs/')
    sd = SaveData()

    round = 10
    accs = []
    for i in range(round):
        print('Round {}/{} modelling:'.format(i+1,round))
        recorder_dict = main(i,sd,log)
        accs.append(float(recorder_dict['results']['accuracy'][0]))
        print_txt = 'accs: {}, mean: {}, std: {}.'.format(accs,np.mean(accs),np.std(accs))
        print(print_txt)
        log.logger.info(print_txt)

    print("hello world~")