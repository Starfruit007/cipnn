import logging 
import time,os

import pickle

class CreateLogger:
    def __init__(self,log_path='logs'):
        self.log_path = log_path
        self.logger = self.create_logger()

    def create_logger(self):
        log_path = self.log_path
        logger = logging.getLogger()
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.setLevel(logging.INFO) 

        rq = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) 
        log_name = os.path.join(log_path, rq + '.log')
        logfile = log_name
        
        fh = logging.FileHandler(logfile, mode='a', encoding = 'utf-8') 
        fh.setLevel(logging.DEBUG) 

        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        logger.info('\n\n\n-------------------------------------------------------------')
        logger.info('this is a logger info message')

        return logger

    def add_logger_info(self, args=None):
        logger = logging.getLogger()
        log_txt = "Configured parameters: \n"
        for argname in vars(args):
            log_txt = log_txt + argname + ': ' + str(getattr(args,argname)) + '\n'
        logger.info(log_txt)



class SaveData:
    def __init__(self) -> None:
        self.recorder_all = {}
        
    def SavePickle(self,data, filePath):
        dataOutput = open(filePath, 'wb')
        pickle.dump(data, dataOutput, 4)
        dataOutput.close()

    def OpenPickle(self,filePath):
        dataInput = open(filePath, 'rb')
        data = pickle.load(dataInput)
        dataInput.close()
        return data

    def main(self,recorder_dict, filePath = 'results.pkl'):
        for ky in recorder_dict:
            if ky not in self.recorder_all: self.recorder_all[ky] = []
            self.recorder_all[ky].append(recorder_dict[ky])
        self.SavePickle(self.recorder_all,filePath)
        print('results saved.\n')