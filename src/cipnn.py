import torch

class CIPNN(torch.nn.Module):
    ''' 
        Random Variable Explanation
            Input sample: Xk
            Model outputted random variables: Z = (z1, z2, ..., zN)
            Label of input sample: Yl
    '''
    def __init__(self, model_type = 'cipnn',forget_num = 2000, stable_num =  1e-20,\
                 scaler_factor = 4.13273, monte_carlo_num = 2, gamma = 0.9, beta = 1, num_params = 2):
        super().__init__()

        self.forget_num = forget_num
        self.stable_num = stable_num
        self.monte_carlo_num = monte_carlo_num
        self.gamma = gamma
        self.beta = beta
        self.num_params = num_params
        self.scaler_factor = scaler_factor

        self.recorder = {'params_T':{},'y_true_T':{},'probability':{}}

        self.distribution = torch.distributions.Normal(0, 1)
        if model_type == 'cipae':
            self.loss_func = torch.nn.BCELoss() # reduction='sum'
        self.model_type = model_type


    def forward(self, logits, y_trues = None, select_variables = None):
        '''
            calculation of the loss and the inference probability $P^{Z}(Yl|Xk)$ with posterior.
            Args:
                logits: output nodes of neural network as paramters of some distribution, such as Gaussian distribution.
                y_trues: list - multi-degree classification
                    label of input samples in a mutli-degree level.
                select_variables: sub joint space selection indexes, which corresponds to y_trues. 
                    If it is set to None, the multi-degree classification will not perform.

            shape symbols:
                b --> batch size                       = logits.shape[0]
                n --> dimension of latent variables    = logits.shape[-1] / num_params
                t --> forget number                    = self.forget_num
                m --> Monte Carlo number               = self.monte_carlo_num
                y --> number of classification classes = y_trues[i].shape[-1]
        '''

        # Take, Gaussian distribution for example, the model outputted logits is the parameters $\mu, \sigma$.
        # Therefore, the num_params is set to 2.
        # Reshape process: [b,n*2] --> [b,n,2]
        params = torch.reshape(logits,[logits.shape[0],-1,self.num_params])

        losses = []
        latent_vars = []
        # for loop is used for multi-degree classification task
        if select_variables is None: select_variables = [list(range(params.shape[1]))]   
        for i in range(len(select_variables)):
            if y_trues[i] is not None:
                params_T,y_true_T = self.parameter_recorder(i, params,y_trues[i]) # shape: [t,n,2]
            else:
                params_T,y_true_T = self.recorder['params_T'][i], self.recorder['y_true_T'][i]  # shape: [t,n,2]

            # for multi degree classificaiton (or clustering) task, use predefined random variables.
            select_params = params[:,select_variables[i],:] # shape: [b,n,2] and n = len(select_variables[i])
            select_params_T = params_T[:,select_variables[i],:] # shape: [t,n,2] and n = len(select_variables[i])


            # random samples and reparamterization trick to get latent variable: Z = (z1,z2,...,zN)
            # shape: [b,m,n]
            z = self.gaussian_sample_and_reparameterize(select_params)

            # oberservation and inference to get the posterior: $P^{Z}(Yl \mid Xk)$
            # shape: [b,y]
            self.recorder['probability'][i], _ = self.observation_and_inference(z, select_params_T,y_true_T)


            # cross entropy loss
            if y_trues[i] is not None:
                if self.model_type == 'cipnn':
                    probs_sum = torch.sum(torch.mul(self.recorder['probability'][i],y_trues[i]),dim = 1) # shape: [b]
                    main_loss = torch.sum(torch.sum(-torch.log(probs_sum),dim = -1)) / logits.shape[0]
                elif self.model_type == 'cipae':
                    main_loss = self.loss_func(self.recorder['probability'][i],y_trues[i])  # logits.shape[0] #/ (y_trues[i].shape[0]*y_trues[i].shape[1])
                losses.append(main_loss)

            latent_vars.append(z)

        # regularization term to organize the joint latent variables space.
        if y_trues[0] is not None:
            kl_loss = self.KL_loss(params, gamma = self.gamma, beta = self.beta)
            losses.append(kl_loss)
        
        return dict(params = params,
                    losses = losses,
                    probability = self.recorder['probability'],
                    latent_vars = latent_vars,)
    
    def de_params(self,params):
        mean, log_var = params[:,:,0],params[:,:,1] # shape: [b,n] or [b,t]
        return mean, log_var

    def gaussian_sample_and_reparameterize(self, params):
        ''' random samples from normal gaussian distribution (noise distribution)
            and reparameterization trick to have the algorithm differentiable'''
        mean,log_var = self.de_params(params) # shape: [b,n]
        # random samples
        eps = self.distribution.sample([self.monte_carlo_num] + list(log_var.shape)).to(mean.device)  # shape: [m,b,n]
        std = torch.exp(log_var/2)

        # reparameterization trick.
        z = mean + eps * std
        z = torch.permute(z,(1,0,2)) # shape: [m,b,n] --> [b,m,n]
        return z

    def observation_and_inference(self,z,select_params_T,y_true_T):
        ''' oberservation and inference to get the posterior: $P^{Z}(Yl \mid Xk)
            Because the posterior is formulated as expectation and uses monte carlo method to approXkmate it,
            the observation and inference phase calculation is then not able to be sperated like the algorithm in IPNN. '''
        
        # input shape: latent_vars - [b,m,n], select_params_T - [t,n]
        # output shape: [b,m,t]
        joint_probs = self.product_gaussian(z,select_params_T)
        
        num_y_joint = torch.einsum('bmt,ty->bmy',joint_probs,y_true_T) # shape: [b,m,y]
        num_joint = torch.unsqueeze(torch.einsum('bmt->bm',joint_probs),dim=-1) # shape: [b,m,1]

        probs = torch.clamp_min(num_y_joint,self.stable_num)/torch.clamp_min(num_joint,self.stable_num) # shape: [b,m,y]

        probability = torch.sum(probs,dim=1) / num_joint.shape[1] # shape:  [b,m,y] -->  [b,y]

        probability = torch.clamp(probability,0,1)

        return probability, (num_y_joint,num_joint)


    def product_gaussian(self, z, params):
        ''' substitute latent variable z into gausstion distribution function to get the probability. '''
        mean,log_var = self.de_params(params) # shape: [t,n]
        var = torch.clamp_min(torch.exp(log_var),1e-20)
        z_ = torch.unsqueeze(z,dim=-2) # shape: [b,m,n] --> [b,m,1,n]
        
        # substitute z value into gaussian distribution function to get the probability.
        # shape of z_ - mean: [b,m,1,n] - [t,n] --> [b,m,t,n]
        p = 1/torch.sqrt(2*torch.pi*var) * torch.exp(-torch.square(z_-mean)/(2*var))

        # expectation scaler factor for large latent space.
        p *= self.scaler_factor

        joint_probs = torch.prod(p,dim=-1) # shape: [b,m,t,n] --> [b,m,t]

        joint_probs = torch.clamp_max(joint_probs,1e20) # avoid inf value.

        return joint_probs

    def KL_loss(self,params, gamma = 0.9, beta = 1):
        ''' regularization term to organize the joint latent variables space
            in order to avoid overfitting problem.'''
        mean,log_var = self.de_params(params) # shape: [b,n]

        # batch mean of kl for each latent dimension
        latent_kl = 0.5 * (-1 - log_var + (1-gamma).pow(2) * mean.pow(2) + log_var.exp()).mean(dim=0)  # shape: [n]
        total_kl = latent_kl.sum()

        return total_kl * beta  

    def parameter_recorder(self, i=None, params=None,y_true=None):
        ''' record labels and parameters of prior distribution, such as gaussian distribution.
            And forget the previous observations according to forget number: self.forget_num.
            
            Args:
                i: indicator of multi-degree classification task.
                num_y_joint: numerator of conditional probability P(Yl|z1,z2,...,zN)
                num_joint: denominator of conditional probability P(Yl|z1,z2,...,zN)
        '''
        # record the previous parameters into recorder.
        if not i in self.recorder['params_T']:
            self.recorder['params_T'][i] = params.detach().clone()
            self.recorder['y_true_T'][i] = y_true.detach().clone()
        else:
            self.recorder['params_T'][i] = torch.vstack((self.recorder['params_T'][i],params.detach().clone()))
            self.recorder['y_true_T'][i] = torch.vstack((self.recorder['y_true_T'][i],y_true.detach().clone()))

        # forgetting: if the memory list length is higher than forget number, then starting forgetting the past observation.
        if self.recorder['params_T'][i].shape[0] > self.forget_num:
            self.recorder['params_T'][i]  = self.recorder['params_T'][i][-self.forget_num:]
            self.recorder['y_true_T'][i]  = self.recorder['y_true_T'][i][-self.forget_num:]


        return self.recorder['params_T'][i], self.recorder['y_true_T'][i] 
