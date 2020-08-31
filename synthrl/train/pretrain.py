import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from random import choices
from tqdm import tqdm

from synthrl.common.environment.dataset import Dataset
from synthrl.common.language.bitvector.lang import BitVectorLang
from synthrl.common.value.bitvector import BitVector
from synthrl.common.value.bitvector import BitVector16
from synthrl.common.value.bitvector import BitVector32
from synthrl.common.value.bitvector import BitVector64
from synthrl.common.function.rnn import RNNFunction
from synthrl.common.environment.dataset import ProgramDataset
from torch.utils.data import DataLoader


# def DataLoader(dataset=None, sample_size=10, io_number=5 , seed=None):
#     if dataset == None:
#         dataset = OracleSampler(sample_size =sample_size, io_number=io_number,seed=seed)
#     programs = []
#     IOs      = []
#     for data in dataset:
#         programs.append(data.program)
#         inputs =[]
#         outputs = []
#         for io in data.ioset:
#             (input, output) = io
#             (param0, param1) = input
#             if not isinstance(param0, BitVector16):
#                 param0 = BitVector16(param0)
#             if not isinstance(param1, BitVector16):
#                 param1 = BitVector16(param1)
#             input = (param0, param1)
#             if not isinstance(output, BitVector16): 
#                 output = BitVector16(output)
#             inputs.append(input)
#             outputs.append(output)
#         IOs.append((inputs,outputs))
#     return programs, IOs


def RnnInit(seq_len, batch_size, n_example, device=None,hidden_size=256):
    hn= torch.zeros(1,batch_size* n_example, hidden_size)
    cn = torch.zeros(1,batch_size*n_example, hidden_size)
    outputs = torch.zeros(batch_size*n_example, seq_len, hidden_size)
    if not (device == None):
        hn =hn.to(device)
        cn =cn.to(device)
        outputs = outputs.to(device)
    hidden = (hn,cn)
    return hidden, outputs



# def PreTrain(agent, programs, IOs, epochs=100):
#     print("Pretraining Start")
#     print("Epochs: {}".format(epochs))
#     print("Sampled Pgms: {}".format(len(programs)))
#     print("Sampled IOs for each Pgm: {}".format(len(IOs[0][0])))
#     optimizer = optim.Adam(list(model.parameters()) + list(emb_model.parameters()) )
#     pre_train_losses = []

#     for epoch in tqdm(range(epochs)):
#         loss = torch.zeros(1)
#         loss = loss.to(device)
#         for idx, prog in enumerate(programs):
#             optimizer.zero_grad()
#             batch_size = 1
#             n_example = len(IOs[idx][0])
#             tokenized = prog.tokenize()
#             (io_inputs,io_outputs) = IOs[idx]

#             for k in range(len(tokenized)):#iterating each partial programs
#                 if k == 0:
#                     hidden, outputs = RnnInit(seq_len, batch_size, n_example, device,hidden_size=256)
#                     emb_val = emb_model( ["HOLE"], [io_inputs], [io_outputs] )
#                     emb_val = emb_val.to(device)
#                     y_policy, y_value, query, hidden = model(emb_val, hidden, outputs)
#                 else:
#                     hidden, outputs = RnnInit(seq_len, batch_size, n_example, device,hidden_size=256)
#                     for token in tokenized[:k]:
#                         emb_val = emb_model( [token], [io_inputs], [io_outputs])
#                         emb_val= emb_val.to(device)
#                         y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
#                         query = query.permute(1,0,2)
#                         outputs = torch.cat((outputs,query),dim=1)
#                 idx_a_t = sorted(BitVectorLang.tokens).index(tokenized[k])
#                 loss+=(y_policy[0][idx_a_t].float().log())
        
#         loss = - torch.div(loss,len(programs))
#         print("Loss at epoch {} is {}".format(epoch+1, loss[0]))
#         pre_train_losses.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print("Epoch Prod. {} finished".format(epoch))
    
#     print("FINAL RESULT::" , pre_train_losses)
#     path = "./saved_models/model_pretrain" + ".tar"
#     torch.save({'model_state_dict': model.state_dict() , 'emb_model_state_dict': emb_model.state_dict() }, path)
#     print("Model Saved!!")

def PreTrain(agent, dataset, epochs):
    print("Pretraining Start")
    print("Epochs: {}".format(epochs))

    optimizer = optim.Adam(list(agent.parameters()))
    pre_train_losses = []
    
    for epoch in range(epochs):
        states = dataset.states[:10]
        agent.evaluate(states)
if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    if torch.cuda.is_available():
        print("GPU ", torch.cuda.get_device_name(0))
    
    token_dim, value_dim, hidden_size = 15, 40, 15
    epochs = 5

    synth = RNNFunction(BitVectorLang,token_dim, value_dim, hidden_size, 1, device)
    dataset = ProgramDataset(dataset_paths=["../common/dataset/train/train_dataset_uptolv01.json"])
    PreTrain(synth, dataset, epochs)
    
    # # for lv in ["01","2","3","4","5"]
    # print("Training Level {} start".format(lv))
    # epochs = 100 * int(lv)
    # dataset = Dataset.from_json("train_dataset_uptolv" + lv + ".json")
    # programs, IOs = DataLoader(dataset=dataset, sample_size = None, io_number =None, seed=None)
    # PreTrain(emb_model, model, programs, IOs ,epochs)    
    # print("Training finished.")
    
    # chk = torch.load("./saved_models/model_pretrain.tar")
    # model.load_state_dict(chk['model_state_dict'])
    # emb_model.load_state_dict(chk['emb_model_state_dict'])


