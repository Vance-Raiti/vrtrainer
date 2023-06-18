import time
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.cuda.amp import autocast, GradScaler

import torch.optim as optim


class Writer():
    @abstractmethod
    def record(self,model_output):
        pass
    
    @abstractmethod
    def write(self):
        pass


class trainerConfig():
    def __init__(self,**kwargs):
        '''
        parameters for optimizing computations and quality-of-life
        args:
            ckpt_pth: path to save model parameters to
            print_interval: number of iterations between prints during training
            save_interval: number of iterations between saves during training
            batch_size: batch size to use for dataloader objects
            dloader_workers: number of workers to use for dataloader objects
            print_function: takes the model's output and produces 
        '''
        self.ckpt_pth = None
        self.max_print_interval = None
        self.save_interaval = None
        self.batch_size = 32
        self.dloader_workers = 0
        self.writer = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for k,v in kwargs.items():
            setattr(self, k, v)
        
        if not (self.writer is None or isinstance(self.writer,Writer)):
            raise TypeError(f"trainerConfig.writer must inherit from trainer.Writer (got {type(self.writer)})")


class Trainer():
    def __init__(self,
                model: torch.nn.Module,
                config: trainerConfig,
                optimizer: optim.Optimizer = None,
                loss_fn = None,
		train_data: Dataset = None,
                test_data: Dataset = None,
                epochs: int = None
            ):
        '''
            args:
                model - instance of model to train
                optimizer - instance of optimizer
                train_data - iterable dataloder object whose __getitem__() method
                    will return a (features, targets) tuple
                config - instance of vrtrainer.trainerConfig. Contains parameters
                    for quality-of-life and performance
		loss_fn - instance of loss function.
        '''
        self.model = model.to(config.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.epochs = epochs


    def train(self):
        self.current_print_interval = 1
        assert self.train_data is not None, "Must have included train data to call train()"
        self.train()

        for epoch in range(self.config.epochs):
            run_epoch('train')
            
    def run_epoch(self,mode):
        model = self.model
        last_time = time.monotonic()
        dataset = self.train_data if mode=='train' else self.test_data
        data = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers = self.config.num_workers
        )
        optimizer = self.optimizer()

        if device=='cuda':
            scaler = Scaler()
            context = autocast
        else:
            context = nocontext   
 
        assert mode == 'train' or mode == 'test', "epochs can only train or test"

        for it, (features, targets) in enumerate(data):

            #book keeping
            current_time = time.monotonic()
            delta_time = current_time - last_time
            last_time = current_time          

            #forward pass
            features = features.to(config.device)
            targets = targets.to(config.device)
            model_output = model(features)
            if not isinstance(model_output,dict):
                raise TypeError(f"model passed to trainer must produce dictionary output (got {type(model_output})")
            y_hat = model_output["y_hat"]
            with context():
                y_hat = model(features)
                loss = self.loss_fn(y_hat,targets)

            #bookkeeping
            self.config.writer.record(model_output)
            if self.config.max_print_interval is not None and it%self.current_print_interval==0:
                self.config.writer.write()
                self.current_print_interval *= 2
                if self.current_print_interval > self.config.max_print_interval:
                    self.current_print_interval = self.config.max_print_interval
            
            #backward pass
            if mode == 'train':       
                optimizer.zero_grad()
                if isinstance(context,nocontext): 
                    loss.backward()    
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if self.config.save_interval is not None and it%self.config.save_interval==0:
                    pth = self.config.ckpt_pth+f"-{it}.pt"
                    torch.save(model,pth)
 
                    
   if config.ckpt_pth is not None and mode=='train':
        torch.save(model,config.ckpt_pth+".pt")
    
    def test(self):
        self.current_print_interval = 1
        assert self.test_data is not None, "Must have included test data to perform test"
        run_epoch('test')
        
class nocontext:
    def __enter__(self):
        pass
    def __exit__(self,*args):
        pass       
