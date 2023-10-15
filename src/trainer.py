import torch
from src.utils import CIPNNEvaluation
import logging 
logger = logging.getLogger()

class Exp_Trainer():

    def __init__(self,model,train_loader,eval_loader,num_epoch,learning_rate, weight_decay) -> None:
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.cipnnevl = CIPNNEvaluation()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.learning_rate, weight_decay = self.weight_decay)


    def train_one_epoch(self,epoch):

        for step, (x, labels) in enumerate(self.train_loader):

            loss, outputs = self.model(x,labels)
            results = self.cipnnevl.main(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            act_acc = results['actual_accuracy'][0]
            acc =  results['accuracy'][0]
            average_loss = results['average_loss']
            print_txt = f'\r epoch {epoch+1}/{self.num_epoch} - curr/avg acc: {act_acc}/{acc}- curr/avg loss: {loss.item():>7f}/{average_loss[0]}->main/KL:[{average_loss[1]},{average_loss[2]}], [{step+1:>5d}/{len(self.train_loader):>5d}]'
            print(print_txt, end='')
        print('\n')
        logger.info(print_txt)



    def test_one_epoch(self):

        for step, (x, labels) in enumerate(self.eval_loader):
            with torch.no_grad():
                _, outputs = self.model(x,labels = None)
                results = self.cipnnevl.main(outputs, labels)
                
                act_acc = results['actual_accuracy'][0]
                acc = results['accuracy'][0]
                print_txt = f'\r prediction - curr/avg acc: {act_acc}/{acc}, [{step+1:>5d}/{len(self.eval_loader):>5d}]'
                print(print_txt, end='')
        print('\n')
        logger.info(print_txt)
        return results

    def exp_start(self):
        total_losses = []
        total_accuracy = []
        for epoch in range(self.num_epoch):
            self.cipnnevl.reset_recorder()
            self.train_one_epoch(epoch)

            total_losses.append(self.cipnnevl.recorder_dict['total_losses'])
            total_accuracy.append(self.cipnnevl.recorder_dict['results_all']['actual_accuracy'])

            self.cipnnevl.reset_recorder()
            results = self.test_one_epoch()


        recorder_dict = self.cipnnevl.recorder_dict
        recorder_dict['results'] = results
        recorder_dict['total_losses'] = total_losses
        recorder_dict['total_accuracy'] = total_accuracy

        if 'cipnn' in dir(self.model):
            recorder_dict['cipnn']  = self.model.cipnn.recorder['params_T'][0].to('cpu'), self.model.cipnn.recorder['y_true_T'][0].to('cpu')
        if 'cipae' in dir(self.model):
            recorder_dict['cipae'] = self.model.cipae.recorder['params_T'][0].to('cpu'), self.model.cipae.recorder['y_true_T'][0].to('cpu')

        return recorder_dict
