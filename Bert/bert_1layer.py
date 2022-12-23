
from utlis import *

from sklearn.metrics import matthews_corrcoef


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from transformers import BertAdam, BertForSequenceClassification
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

class bert_1layer():

    def __init__(self,batch_size=32,epochs = 4,load_path=False):


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        self.model.to('cuda')
        #model.cuda() # print the architecture of the model


        # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
        self.batch_size = batch_size
        # Number of training epochs (authors recommend between 2 and 4)
        self.epochs = epochs

        # Store our loss and accuracy for plotting
        self.train_loss_set = []

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]

        # This variable contains all of the hyperparemeter information our training loop needs
        self.optimizer = BertAdam(self.optimizer_grouped_parameters,
                            lr=2e-5,
                            warmup=.1)

        self.model.load_state_dict(torch.load(load_path))



    def fit(self,data):

        train_inputs, train_masks, train_labels,validation_inputs, validation_masks, validation_labels = preprocess_train(data)
        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
        # with an iterator the entire dataset does not need to be loaded into memory

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)

        # trange is a tqdm wrapper around the normal python range
        for _ in trange(self.epochs, desc="Epoch"):
        
        
            # Training
            
            # Set our model to training mode (as opposed to evaluation mode)
            self.model.train()
            
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(self.device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Clear out the gradients (by default they accumulate)
                self.optimizer.zero_grad()
                # Forward pass
                loss = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                self.train_loss_set.append(loss.item())    
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                self.optimizer.step()
                
                
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss/nb_tr_steps))
                
                
            # Validation

            # Put model in evaluation mode to evaluate loss on the validation set
            self.model.eval()

            # Tracking variables 
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(self.device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

    def plot_loss(self):
        plt.figure(figsize=(15,8))
        plt.title("Training loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.plot(self.train_loss_set)
        plt.show()

    
    def test_evaluate(self,data):

        prediction_inputs, prediction_masks, prediction_labels = preprocess_test(data)

        prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)


        # Prediction on test set

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables 
        predictions , true_labels = [], []

        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        
        # Import and evaluate each test batch using Matthew's correlation coefficient

        matthews_set = []

        for i in range(len(true_labels)):
            matthews = matthews_corrcoef(true_labels[i],
                            np.argmax(predictions[i], axis=1).flatten())
            matthews_set.append(matthews)

        # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in true_labels for item in sublist]

        print(matthews_corrcoef(flat_true_labels, flat_predictions))


















