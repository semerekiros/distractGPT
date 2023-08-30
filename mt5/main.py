import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from random import randrange
from utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
import json
import time
import csv
import tqdm
import numpy as np
import wandb
import pandas as pd
import math
from datetime import datetime
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nlgeval import NLGEval
from utils import exceed_512_
from torch import cuda
from my_dataset import Race_Dataset, Race_Dataset_test
device = 'cuda' if cuda.is_available() else 'cpu'

# WandB – Import the wandb library


def train(epochs, tokenizer, model, device, loader, val_loader, optimizer, save_path):

    for epoch in epochs:

        model.train()

        total_loss = 0.
        best_val_loss = 1000000000000000.0
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            total_loss+= loss.item()

            #if _ % 10 == 0:
            #    wandb.log({"Training Loss per 10 steps": loss.item()})

            if _ % 500 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #After the finish of epoch remember to save the checkpoint
        #run_string = datetime.now().strftime("%m_%d_%Y_%H_%M")
        run_string = save_path + "/" + str(epoch)
        output_path = "checkpoints/"+str(epoch + 1)+".pt"
        output_path_ = "checkpoints/" + run_string

        #torch.save(model.state_dict(), output_path)
        #model.save_pretrained(output_path_)

        #print(f'Epoch: {epoch}, Perplexity:  {math.exp(total_loss)}')
        print(f'Epoch: {epoch}, Total_loss:  {total_loss}')

        val_loss, val_perplexity = evaluation(epoch, tokenizer, model, device, val_loader)
        if val_loss < best_val_loss:
            #output_path_ = save_path + "/" + str("bestmodel")
            model.save_pretrained("checkpoints/best_model.pt")
            best_val_loss = val_loss

        wandb.log({"Training total Loss": total_loss, "Training perplexity":10})
        wandb.log({"Validation total Loss": val_loss, "Validation perplexity": val_perplexity})




def evaluation (epoch, tokenizer, model, device, loader):
    val_loss = 0.
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            val_loss+=loss.item()
    val_perplexity = math.exp(val_loss)
    val_perplexity = 10

    return val_loss, val_perplexity

def validate_decode(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)

    return predictions, actuals


def main():
    # WandB – Initialize a new run
    wandb.init(project="fine_tune_mt5_televic_dist_gen")

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training
    config = wandb.config  # Initialize config
    config.TRAIN_BATCH_SIZE = 10  # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 100  # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 10  # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1
    config.LEARNING_RATE = 1e-5  # learning rate (default: 0.01)
    config.SEED = 42  # random seed (default: 42)
    config.MAX_LEN = 512
    config.TARGET_LEN = 150
    config.CHECKPOINT = 'google/mt5-xxl'

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    #tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained(config.CHECKPOINT)    #mt5 checkpoint     google/mt5-xxl, google/mt5-large, google/mt5-base

    train_path = "./test-data/train.json"
    test_path = "./test-data/test.json"
    val_path = "./test-data/valid.json"

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = Race_Dataset(train_path, tokenizer, config.MAX_LEN, config.TARGET_LEN)
    val_set = Race_Dataset(val_path, tokenizer, config.MAX_LEN, config.TARGET_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.

    #model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained(config.CHECKPOINT)

    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on the Televic dataset')
    print(f'using the checkpoint {config.CHECKPOINT}')
    save_path = datetime.now().strftime("%m_%d_%Y_%H_%M")

    train(config.TRAIN_EPOCHS, tokenizer, model, device, training_loader, val_loader,optimizer, save_path)

    #for epoch in range(config.TRAIN_EPOCHS):
     #   train(epoch, tokenizer, model, device, training_loader, val_loader,optimizer, save_path)


    '''
    # Validation loop and saving the resulting file with predictions and actual in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating distractors on our fine tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(config.VAL_EPOCHS):
        predictions, actuals = validate_decode(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
        final_df.to_csv('./models/predictions.csv')
        print('Output Files generated for review')

    model2 = T5ForConditionalGeneration.from_pretrained("checkpoints/")
    model2 = model2.to(device)
    for epoch in range(config.VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model2, device, val_loader)
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
        final_df.to_csv('./models/predictions_saved.csv')
        print('Output Files generated for review')

'''

if __name__ == '__main__':
    main()
