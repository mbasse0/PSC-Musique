from model import *
from model4out import *
from dataset import *
import pytorch_lightning as pl
from csv_encoder import *

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import sys

def main(argv):
    """
    Ne s'utilise que pour le hyperparameter tuning, donc 
        - uniquement pour train
        - sans sauver les modèles
    L'objectif est uniquement d'obtenir les hyperperameters optimaux, pour ensuite train avec main.py
    
    arg0 : 0 : noDDP , 1 : DDP
    arg1 : nb_epochs
    arg2 : dataset_name
    
    """
    dataset_path = "./Datasets/" + argv[2]
    input_vect, rep_vect = tokensFileToVectInputTarget(dataset_path,120)

    def train_jazzbot(config, data_dir=None, num_epochs=1, num_gpus=1):
        model = Transformer(
            num_tokens=len(custom_vocab), 
            dim_model=512, 
            num_heads=8, 
            num_encoder_layers=1, 
            num_decoder_layers=4, 
            dropout_p=0.1, 
            learning_rate=config["lr"]
            )

        train_dataloader, val_dataloader = get_two_dataloaders(input_vect, rep_vect, batch_size=config["batch_size"])
        
        metrics = {"loss": "ptl/val_loss"}
        
        if (int(argv[0]) == 1):
            trainer = pl.Trainer(max_epochs=num_epochs, gpus=num_gpus, strategy='ddp', accelerator='gpu', benchmark=True, progress_bar_refresh_rate=0, callbacks=[TuneReportCallback(metrics, on="validation_end")])
        else:
            trainer = pl.Trainer(max_epochs=num_epochs, gpus=num_gpus, accelerator='gpu', benchmark=True, progress_bar_refresh_rate=0, callbacks=[TuneReportCallback(metrics, on="validation_end")])
    
        trainer.fit(model, train_dataloader, val_dataloader)
  
    num_samples = 10
    num_epochs = int(argv[1])
    gpus_per_trial = 3 if (int(argv[0]) == 1) else 1

    config = {
        "lr": tune.loguniform(1e-3, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64]),
    }

    trainable = tune.with_parameters(
        train_jazzbot,
        #data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial)

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        name="tune_jazzbot")

    print(analysis.best_config)


if __name__=="__main__":
   main(sys.argv[1:])
