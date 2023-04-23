from vocab import *
from model import *
from model4out import *
import numpy as np
from generate import *
from data_encoder import *
from dataset import *
import pytorch_lightning as pl
from config import *
import sys
from csv_encoder import *

from ray.tune.integration.pytorch_lightning import TuneReportCallback
import tempfile
from ray import tune

def main():

    dataset_path = "./Datasets/"
    input_vect, rep_vect = tokensFileToVectInputTarget(dataset_path,120)

    def train_jazzbot(config, data_dir=None, num_epochs=10, num_gpus=3):
        model = Transformer(
            num_tokens=len(custom_vocab), 
            dim_model=512, 
            num_heads=8, 
            num_encoder_layers=1, 
            num_decoder_layers=4, 
            dropout_p=0.1, 
            learning_rate= 0.05
            )

        train_dataloader, val_dataloader = get_two_dataloaders(input_vect, rep_vect, batch_size=config["batch_size"])
        
        metrics = {"loss": "ptl/val_loss"}
        
        trainer = pl.Trainer(max_epochs=num_epochs, gpus=num_gpus, progress_bar_refresh_rate=0, callbacks=[TuneReportCallback(metrics, on="validation_end")])
    
        trainer.fit(model, train_dataloader, val_dataloader)
  
    num_samples = 10
    num_epochs = 10
    gpus_per_trial = 3 

    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
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
   main()
