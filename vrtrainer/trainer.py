import time
import torch

class trainerConfig():
    def __init__(self,**kwargs):
        self.epochs = 1
        self.ckpt_pth = 'model'
        self.print_interval = None
        self.save_interaval = None

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        for k,v in kwargs.items():
            setattr(self, k, v)



class Trainer():
    def __init__(self,model,loss_fn,optimizer,train_data,test_data,config: trainerConfig):
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

    def train(self):
        model = self.model
        last_time = time.monotonic()
        running_loss = 0.0

        for epoch in range(self.config.epochs):
            for it, (features, targets) in enumerate(self.train_data):
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
        model = self.model
        running_loss = 0.0

        for features, targets in self.test_data:
            y_hat = model(features)
            running_loss += self.loss_fn(y_hat,targets).item()
    
        return running_loss/len(self.test_data)



                
