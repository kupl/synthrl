import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from random import choices
from tqdm import tqdm
from math import ceil
from synthrl.common.environment.dataset import Dataset
from synthrl.common.language.bitvector.lang import BitVectorLang
from synthrl.common.value.bitvector import BitVector
from synthrl.common.value.bitvector import BitVector16
from synthrl.common.value.bitvector import BitVector32
from synthrl.common.value.bitvector import BitVector64
from synthrl.common.function.rnn import RNNFunction
from synthrl.common.environment.dataset import ProgramDataset
from synthrl.common.environment.dataset import iterate_minibatches
from torch.utils.data import DataLoader

def labels2idx(agent, labels):
    sorted_tokens = sorted(BitVectorLang.TOKENS) #to avoid traumatic situation
    idx_list = [sorted_tokens.index(label) for label in labels]
    idx_list = torch.tensor(idx_list)
    return idx_list


def PreTrain(agent, dataset, batch_size, epochs):
    agent.token_emb.train()
    agent.value_emb.train()
    agent.network.train()

    dataset_len = len(dataset)
    print("Pretraining Start")
    print("Epochs: {}".format(epochs))
    print("Dataset Length: {}".format(dataset_len))

    optimizer = optim.Adam(agent.parameters())
    loss_fn = nn.NLLLoss()
    losses = []

    for epoch in range(epochs):
        loss_by_epoch  = torch.Tensor([0])
        for batch_idx, batch in enumerate(iterate_minibatches(dataset=dataset, batch_size=batch_size, shuffle=True)):
            states, labels = batch
            policies, values = agent.evaluate(states) 
            labels = labels2idx(agent, labels)
            loss = loss_fn(policies.log(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('Loss at Epoch {:4d}/{} Batch {:2d}/{} Cost: {:.6f}'.format(epoch+1, epochs, batch_idx+1,
             int(ceil(dataset_len/batch_size)) ,loss.item()))
            loss_by_epoch+=loss

        if (epoch+1) % 10 == 0:
            path="../saved_models/model_epoch" + str(epoch+1) + ".tar"
            agent.save(path)
            print("Model Saved at Epoch {}".format(epoch+1))
        
        losses.append(loss_by_epoch.item() / int(ceil(dataset_len/batch_size)) )
        print("Loss(avg) at Epoch {} : {}".format(epoch+1,loss_by_epoch.item()/int(ceil(dataset_len/batch_size))))
    print("Pretraining Finshed")
    print("Final Loss(avg) Result:")
    print(losses)

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    if torch.cuda.is_available():
        print("GPU ", torch.cuda.get_device_name(0))

    token_dim, value_dim, hidden_size = 15, 40, 15
    epochs = 20
    batch_size = 256
    
    synth = RNNFunction(BitVectorLang,token_dim, value_dim, hidden_size, 1, device)
    dataset = ProgramDataset(dataset_paths=["../common/dataset/train/train_dataset_uptolv01.json"])
    PreTrain(synth, dataset, batch_size, epochs)

      # paths = ["../dataset/train/train_dataset_uptolv01.json",
  #           "../dataset/train/train_dataset_uptolv2.json",
  #           "../dataset/train/train_dataset_uptolv3.json",
  #           "../dataset/train/train_dataset_uptolv4.json",
  #           "../dataset/train/train_dataset_uptolv5.json" ]
  # paths = ["../dataset/train/train_dataset_uptolv01.json" ] 



