from nets.envs import SCI

import os
import numpy as np
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchmetrics import R2Score
import neptune.new as neptune
import neptune.new.integrations.optuna as outils

DEVICE = torch.device("cpu")
BATCHSIZE = 2
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 10
N_VALID_EXAMPLES = BATCHSIZE * 10

class Hyper(SCI):
    def __init__(self,idx:tuple=(1,3,7),*args, **kwargs):
        super(Hyper,self).__init__()
        self.loader = self.data_flow(idx=idx)
        
    # call self dataflow
    def define_model(self,trial):
    # We optimize the number of layers, hidden units and 
        n_layers = trial.suggest_int("n_layers", 2, 6)
        layers = []

        in_features = self.input_dim
        
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
            activations = trial.suggest_categorical('activation',['ReLU','Tanh','SiLU','SELU','RReLU'])
            
            layers.append(nn.Linear(in_features, out_features))
            layers.append(getattr(nn,activations)())
            p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.2)
            layers.append(nn.Dropout(p))

            in_features = out_features
        layers.append(nn.Linear(in_features, 1))

        return nn.Sequential(*layers)
    def objective(self,trial):
        # Generate the model.
        model = self.define_model(trial).to(DEVICE)
        mape = R2Score()
        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD",'AdamW','Adamax','Adagrad'])
        lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        
        train_loader, valid_loader = self.loader,self.loader

        # Training of the model.
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # Limiting training data for faster epochs.
                if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break

                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = F.mse_loss(output, target)
                loss.backward()
                optimizer.step()

            # Validation of the model.
            model.eval()
            correct = 0
            pred=torch.tensor([])
            targs=torch.tensor([])
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    # Limiting validation data.
                    if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                        break
                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                    
                    output = model(data)
                    # print(output,target)
                    # print(output.squeeze()-target)
                    # # Get the index of the max log-probability.
                    pred = torch.cat((pred,output.squeeze()))
                    targs = torch.cat((targs,target))
                    # correct += pred.eq(target.view_as(pred)).sum().item()
            
            accuracy = mape(pred,targs)

            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy
    
    def start_study(self,n_trials:int=100,neptune_project:str=None,neptune_api:str=None):
        
        
        study = optuna.create_study(direction="maximize")
        if neptune_project and neptune_api:
            run = neptune.init_run(
                project=neptune_project,
                api_token=neptune_api,
            )
            neptune_callback = outils.NeptuneCallback(run)
            study.optimize(self.objective, n_trials=n_trials, timeout=600,callbacks=[neptune_callback])
        else:
            study.optimize(self.objective, n_trials=n_trials, timeout=600)
        
       

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        self.trial = study.best_trial

        print("  Value: ", self.trial.value)

        print("  Params: ")
        for key, value in self.trial.params.items():
            print("    {}: {}".format(key, value))
            
        if neptune_api and neptune_project:
            run.stop()
            
        return {"  Number of finished trials: ":len(study.trials),
                "  Number of pruned trials: ": len(pruned_trials),
                "  Number of complete trials: ": len(complete_trials),
                "Best trial score" : self.trial.value,
                "  Params: ": self.trial.params
        }