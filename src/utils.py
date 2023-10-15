
import numpy as np

class CIPNNEvaluation:
    def __init__(self) -> None:
        self.recorder_dict = {}

    def reset_recorder(self):
        self.recorder_dict = {}
        
    def torch_to_numpy(self, outputs):
        ''' convert type of model outputs from tensor to numpy.'''
        outputs_numpy = {}
        for ky in outputs:
            if isinstance(outputs[ky],list):
                outputs_numpy[ky] = [_.detach().cpu().numpy() for _ in outputs[ky]]
            elif isinstance(outputs[ky],dict):
                outputs_numpy[ky] = {}
                for subky in outputs[ky]:
                    outputs_numpy[ky][subky] = outputs[ky][subky].detach().cpu().numpy()
            elif isinstance(outputs[ky],object):
                outputs_numpy[ky] = outputs[ky].detach().cpu().numpy()

        return outputs_numpy

    def compare(self, preds,label):
        ''' comparison between predictions and labels. '''
        preds_label = np.argmax(preds,axis = -1)
        true_nums = np.sum(preds_label == label)
        return preds_label, true_nums

    def accuracy_calc(self, outputs_numpy, labels_numpy):
        '''accuracy calcation: prediction is the indexes of maximum posterior.
        '''
        true_nums_dict = {}
        preds_label_dict = {}
        for i in outputs_numpy['probability']: # for multi-degree classification task.
            preds = outputs_numpy['probability'][i]
            if i==0 or labels_numpy.shape[-1] == len(list(outputs_numpy['probability'].keys())):
                label = labels_numpy[:,i]
                preds_label, true_nums = self.compare(preds,label)
            elif i > 0: 
                label = labels_numpy[:,0]
                preds_label, true_nums = self.compare(preds,label)
                true_nums = 0
            true_nums_dict[i] = true_nums
            preds_label_dict[i] = preds_label

        return true_nums_dict, preds_label_dict


    def main(self,outputs, labels):
        ''' evaluation results will be returned.
            Besides, some varialbes are recorded in self.recorder_dict for later analysis use.'''
        outputs_numpy = self.torch_to_numpy(outputs)
        labels_numpy = labels.detach().cpu().numpy()

        batch_size = labels_numpy.shape[0]
        if len(labels_numpy.shape)==1: # shape higher than 1 is the case of multi-degree classification task.
            labels_numpy = np.expand_dims(labels_numpy,axis = -1)
            
        true_nums_dict, preds_label_dict = self.accuracy_calc(outputs_numpy, labels_numpy) # actual accuracy calculation of the input batch.
        act_acc = {ky:true_nums_dict[ky]/batch_size for ky in true_nums_dict}


        # record the evaluation results into self.recorder_dict
        losses_numpy = np.array(outputs_numpy['losses'])
        self.recorder(labels_numpy,losses_numpy,preds_label_dict,true_nums_dict,outputs_numpy['latent_vars'])


        total_number = self.recorder_dict['total_labels'].shape[0]
        acc = {ky:self.recorder_dict['total_true_nums_dict'][ky]/total_number for ky in self.recorder_dict['total_true_nums_dict']} # overall accuracy calculation.
        
        if list(losses_numpy) != []:
            average_loss = np.sum(self.recorder_dict['total_losses'],axis = 0) / self.recorder_dict['total_losses'].shape[0]
        else: average_loss = np.zeros(2)
        
        results = dict(
            accuracy = acc,
            actual_accuracy = act_acc,
            average_loss = average_loss
        )
        self.record_results(results) # record evaluation results into self.recorder_dict 
        results = self.format(results) # evaluation results rounding
        return results

    def record_results(self,results):
        ''' record evaluation results. '''
        if 'results_all' not in self.recorder_dict:
            self.recorder_dict['results_all'] = {}
        for ky in results:
            if ky not in self.recorder_dict['results_all']:      self.recorder_dict['results_all'][ky] = []
            if isinstance(results[ky],dict):
                tmp = list(results[ky].values())
            else:
                tmp = results[ky]
            self.recorder_dict['results_all'][ky].append(tmp)

    def recorder(self,labels_numpy,losses_numpy,preds_label_dict,true_nums_dict,latent_vars):
        ''' record some variables for later anlaysis use.'''
        if 'total_labels' not in self.recorder_dict:
            self.recorder_dict['total_labels'] = labels_numpy
        else:
            self.recorder_dict['total_labels'] = np.vstack((self.recorder_dict['total_labels'],labels_numpy))

        if 'total_losses' not in self.recorder_dict:
            self.recorder_dict['total_losses'] = np.expand_dims(losses_numpy,axis=0)
        else:
            self.recorder_dict['total_losses'] = np.vstack((self.recorder_dict['total_losses'],losses_numpy))


        if 'total_preds_label_dict' not in self.recorder_dict:
            self.recorder_dict['total_preds_label_dict'] = {ky:preds_label_dict[ky] for ky in preds_label_dict} # avoid share same memory address
        else:
            for ky in preds_label_dict:
                self.recorder_dict['total_preds_label_dict'][ky] = np.hstack((self.recorder_dict['total_preds_label_dict'][ky],preds_label_dict[ky]))
        
        if 'total_true_nums_dict' not in self.recorder_dict:
            self.recorder_dict['total_true_nums_dict'] = {ky:true_nums_dict[ky] for ky in true_nums_dict} # avoid share same memory address
        else:
            for ky in preds_label_dict:
                self.recorder_dict['total_true_nums_dict'][ky] = self.recorder_dict['total_true_nums_dict'][ky] + true_nums_dict[ky]

        
        if 'total_latent_vars' not in self.recorder_dict:
            self.recorder_dict['total_latent_vars'] = latent_vars
        else:
            self.recorder_dict['total_latent_vars'] = [np.vstack((self.recorder_dict['total_latent_vars'][i],latent_vars[i])) for i in range(len(latent_vars))]

    def format(self, results):
        ''' round the evaluation results '''
        results['actual_accuracy'] = [f'{_:>5f}' for _ in results['actual_accuracy'].values()]
        results['accuracy'] = [f'{_:>5f}' for _ in results['accuracy'].values()]

        average_loss = results['average_loss']
        results['average_loss'] = [f'{np.sum(average_loss):>5f}']
        results['average_loss'].extend([f'{_:>5f}' for _ in average_loss])

        return results
