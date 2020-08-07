import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from random import choices


from synthrl.utils.trainutils import Dataset
from synthrl.utils.trainutils import IOSet
from synthrl.utils.trainutils import Element

from synthrl.agent.rl import Network
from synthrl.language.bitvector.lang import VECTOR_LENGTH
from synthrl.language.bitvector.lang import BitVectorLang
from synthrl.value.bitvector import BitVector
from synthrl.value.bitvector import BitVector16
from synthrl.value.bitvector import BitVector32
from synthrl.value.bitvector import BitVector64
from synthrl.language.bitvector.oracle import OracleSampler
from synthrl.language.bitvector.embedding import Embedding



def DataLoader(sample_size=10, io_number=5 , seed=None):
    dataset = OracleSampler(sample_size =sample_size, io_number=io_number,seed=seed)
    programs = []
    IOs      = []
    for data in dataset:
        programs.append(data.program)
        inputs =[]
        outputs = []
        for io in data.ioset:
            (input, output) = io
            (param0, param1) = input
            if not isinstance(param0, BitVector16):
                param0 = BitVector16(param0)
            if not isinstance(param1, BitVector16):
                param1 = BitVector16(param1)
            input = (param0, param1)
            if not isinstance(output, BitVector16): 
                output = BitVector16(output)

            inputs.append(input)
            outputs.append(output)
        IOs.append((inputs,outputs))
    return programs, IOs


def RnnInit(seq_len, batch_size, n_example, device,hidden_size=256):
    hn= torch.zeros(1,batch_size* n_example, hidden_size)
    hn =hn.to(device)
    cn = torch.zeros(1,batch_size*n_example, hidden_size)
    cn =hn.to(device)
    hidden = (hn,cn)
    outputs = torch.zeros(batch_size*n_example, seq_len, hidden_size)
    outputs = outputs.to(device)
    return hidden, outputs


def PreTrain(emb_model, model, epochs=100):
    
    model.train()
    emb_model.train()

    programs, IOs = DataLoader(10,5)
    seq_len=10
    hidden_size=256

    print("Epochs: {}".format(epochs))
    print("Sampled Pgms: {}".format(len(programs)))
    print("Sampled IOs for each Pgm: {}".format(len(IOs[0][0])))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    if torch.cuda.is_available():
        emb_model = emb_model.to(device)
        model = model.to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(emb_model.parameters()) )

    pre_train_losses = []
    for epoch in range(epochs):
        print(epoch)
        loss = torch.zeros(1)
        loss = loss.to(device)
        for idx, prog in enumerate(programs):
            optimizer.zero_grad()
            batch_size = 1
            n_example = len(IOs[idx][0])
            tokenized = prog.tokenize()
            (io_inputs,io_outputs) = IOs[idx]

            for k in range(len(tokenized)):#iterating each partial programs
                #Handling the first state, "HOLE" state:
                if k == 0:
                    hidden, outputs = RnnInit(seq_len, batch_size, n_example, device,hidden_size=256)
                    emb_val = emb_model( ["HOLE"], [io_inputs], [io_outputs] )
                    emb_val = emb_val.to(device)
                    y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
                    idx_a_t = sorted(BitVectorLang.tokens).index(tokenized[k])
                    loss+=(y_policy[0][idx_a_t].float().log())
                else:
                    hidden, outputs = RnnInit(seq_len, batch_size, n_example, device,hidden_size=256)
                    for token in tokenized[:k]:
                        emb_val = emb_model( [token], [io_inputs], [io_outputs])
                        emb_val= emb_val.to(device)
                        y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
                        query = query.permute(1,0,2)
                        outputs = torch.cat((outputs,query),dim=1)
                    idx_a_t = sorted(BitVectorLang.tokens).index(tokenized[k])
                    loss+=(y_policy[0][idx_a_t].float().log())

        loss = - torch.div(loss,len(programs))
        print("Loss at epoch {} is {}".format(epoch+1, loss[0]))
        pre_train_losses.append(loss.item())
        print("Current Train Losses: ", pre_train_losses)
        loss.backward()
        optimizer.step()
    print("FINAL RESULT::" , pre_train_losses)
    path = "./saved_models/model" + ".tar"
    torch.save({'model_state_dict': model.state_dict() , 'emb_model_state_dict': emb_model.state_dict() }, path)
    print("Model Saved!!")
    

def policy_rollout(io_spec,emb_model, model):
    MAX_MOVE = 100
    pgm = BitVectorLang()
    seq_len = 10
    batch_size = 1
    
    io_inputs, io_outputs = io_spec
    n_example = len(io_inputs)

    for k in range(MAX_MOVE):
        if k==0:
            hidden, outputs = RnnInit(seq_len, batch_size, n_example, device, hidden_size=256)
            emb_val = emb_model( ["HOLE"], [io_inputs], [io_outputs])
            y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
            y_policy.tolist()
            sampled_action = choices(BitVectorLang.tokens, y_policy)
        else:
            hidden, outputs = RnnInit(seq_len, batch_size, n_example, device, hidden_size=256)
            for token in pgm.tokenize():
                emb_val = emb_model( [token], [io_inputs], [io_outputs])
                y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
        pgm.production(sampled_action)
        if pgm.is_complete():
            break
    pass

def reward():
    pass

def Train(emb_model, model, epochs):
    pass
    
if __name__=='__main__':
    emb_model = Embedding(token_dim=15,value_dim=40, type=BitVector16)
    model = Network(emb_model.emb_dim,len(BitVectorLang.tokens))
    epochs = 10
    PreTrain(emb_model, model, epochs)
    # Train(emb_model, model, epochs)