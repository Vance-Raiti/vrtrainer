import time
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.optim as optim




class trainerConfig():
    def __init__(self,**kwargs):
        self.ckpt_pth = 'model'
        self.print_interval = None
        self.save_interaval = None
        self.batch_size = 32
        self.dloader_workers = 0

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        for k,v in kwargs.items():
            setattr(self, k, v)



class Trainer():
    def __init__(self,
                model: torch.nn.Module,
                loss_fn,
                config: trainerConfig,
                optimizer: optim.Optimizer = None,
                train_data: Dataset = None,
                test_data: Dataset = None,
                epochs: int = None
            ):
        '''
            args:
                model - instance of model to train
                loss_fn - instance of loss function that returns a torch scalar
                optimizer - instance of optimizer
                train_data - iterable dataloder object whose __getitem__() method
                    will return a (features, targets) tuple
                config - instance of vrtrainer.trainerConfig
        '''
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.epochs = epochs


    def train(self):
        assert self.train_data is not None, "Must have included train data to call train()"
        model = self.model
        last_time = time.monotonic()
        running_loss = 0.0
        data = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            num_workers = self.config.num_workers
        )
        for epoch in range(self.config.epochs):


            for it, (features, targets) in enumerate(data):
                current_time = time.monotonic()
                delta_time = current_time - last_time
                last_time = current_time          


                y_hat = model(features)
                loss = self.loss_fn(y_hat,targets)
                
                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()

                #bookkeeping
                running_loss+=torch.item(loss)
                if self.config.print_interval is not None and it%self.config.print_interval==0:
                    print(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, running loss {running_loss:0.5f}, it/sec: {1.0/delta_time}")
                    running_loss = 0.0
                
                if self.config.save_interval is not None and it%self.config.save_interval==0:
                    pth = self.config.ckpt_pth+f"-{it}.pt"
                    torch.save(model,pth)
    
    def test(self):
        assert self.test_data is not None, "Must have included test data to perform test"
        model = self.model
        running_loss = 0.0
        data = DataLoader(
            self.test_data,
            batch_size=self.config.batch_size,
            num_workers = self.config.num_workers
        )

        for features, targets in data:
            y_hat = model(features)
            running_loss += self.loss_fn(y_hat,targets).item()
    
        return running_loss/len(self.test_data)



                
