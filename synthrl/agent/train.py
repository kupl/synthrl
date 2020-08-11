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
from synthrl.language.bitvector.lang import ExprNode
from synthrl.language.bitvector.lang import BOPNode

from synthrl.language.bitvector.lang import ConstNode
from synthrl.language.bitvector.lang import ParamNode

from synthrl.value.bitvector import BitVector
from synthrl.value.bitvector import BitVector16
from synthrl.value.bitvector import BitVector32
from synthrl.value.bitvector import BitVector64

from synthrl.language.bitvector.oracle import OracleSampler
from synthrl.language.bitvector.embedding import Embedding


class RollOutError(Exception):

  def __init__(self, *args, **kwargs):
    super(RollOutError, self).__init__(*args, **kwargs)



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


def PreTrain(emb_model, model,programs, IOs, epochs=100):
    
    model.train()
    emb_model.train()

    seq_len=10
    hidden_size=256

    print("Pretraining Start")
    print("Epochs: {}".format(epochs))
    print("Sampled Pgms: {}".format(len(programs)))
    print("Sampled IOs for each Pgm: {}".format(len(IOs[0][0])))
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("FINAL RESULT::" , pre_train_losses)
    path = "./saved_models/model_pretrain" + ".tar"
    torch.save({'model_state_dict': model.state_dict() , 'emb_model_state_dict': emb_model.state_dict() }, path)
    print("Model Saved!!")
    

def policy_rollout(io_spec, emb_model, model):
    #The io_spec must be packed by BitVector cls
    MAX_MOVE = 100 
    pgm = BitVectorLang()
    seq_len = 10
    batch_size = 1
    io_inputs, io_outputs = io_spec
    n_example = len(io_inputs)
    for k in range(MAX_MOVE):
        if k==0:
            hidden, outputs = RnnInit(seq_len, batch_size, n_example, device=None, hidden_size=256)
            emb_val = emb_model( ["HOLE"], [io_inputs], [io_outputs])
            y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
        else:
            hidden, outputs = RnnInit(seq_len, batch_size, n_example, device=None, hidden_size=256)
            for token in pgm.tokenize():
                emb_val = emb_model( [token], [io_inputs], [io_outputs])
                y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
        y_policy_li=(y_policy.tolist())[0]
        sampled_action = choices(BitVectorLang.tokens, y_policy_li)
        sampled_action=sampled_action[0]
        if sampled_action in ExprNode.tokens: #ExprNode.tokens=["neg","arith-neg"], by its way of construction
            pgm.production(sampled_action)
        elif sampled_action in BOPNode.tokens:
            pgm.production("bop")
            pgm.production(sampled_action)
        elif sampled_action in ConstNode.tokens:
            pgm.production("const")
            pgm.production(int(sampled_action))
        elif sampled_action in ParamNode.tokens:
            pgm.production("var")
            pgm.production(sampled_action)
        else: 
            print("Something bad happended")
        if pgm.is_complete():
            break
        if (k == MAX_MOVE-1) and ( not pgm.is_complete() ) :
            raise RollOutError("Rollout synthesis couldn't be done in {} moves".format(MAX_MOVE))
    return pgm

def reward(io_spec, prog):
    io_inputs, io_outputs = io_spec
    for idx, io_input in enumerate(io_inputs):
        if prog.interprete(io_input) !=  io_outputs[idx]:
            return 0
    return 1 

def Train(emb_model, model, IOs, epochs):
    model.train()
    emb_model.train()
    batch_size = 1
    optimizer = optim.Adam(list(model.parameters()) + list(emb_model.parameters()) )
    seq_len = 10 
    train_losses = []
    eps=1e-5


    print("Main Training Start")
    for epoch in range(epochs):
        total_avg_loss = 0
        success_rollout = 0 #counts policy rollouts of completed sythesis  
        for io_spec in IOs:
            loss = torch.zeros(1) #According to our paper, "Given the specific rollout, we train v and π to maximize~", So set loss zero val for each specific roll out(i.e for each speicific io_spec)
            
            try:
                prog = policy_rollout(io_spec, emb_model, model)
            except RollOutError:
                continue

            n_example = len(io_spec[0])
            tokenized = prog.tokenize()
            R = reward(io_spec, prog)
            io_inputs, io_outputs = io_spec
            for k in range(len(tokenized)):#iterating each partial programs
                #Handling the first state, "HOLE" state:
                if k == 0:
                    hidden, outputs = RnnInit(seq_len, batch_size, n_example, device = None ,hidden_size=256)
                    emb_val = emb_model( ["HOLE"], [io_inputs], [io_outputs] )
                    y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
                else:
                    hidden, outputs = RnnInit(seq_len, batch_size, n_example, device = None ,hidden_size=256)
                    for token in tokenized[:k]:
                        emb_val = emb_model( [token], [io_inputs], [io_outputs])
                        y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
                        query = query.permute(1,0,2)
                        outputs = torch.cat((outputs,query),dim=1)      
                if R == 1:
                    loss += ( (y_value.float() - eps ).log()).reshape(1)
                    idx_a_t = sorted(BitVectorLang.tokens).index(tokenized[k])
                    loss+=(y_policy[0][idx_a_t].float().log())
                elif R==0:
                    loss +=  ( (1 -  y_value.float()) + eps ).log().reshape(1)
            loss = - loss #Since the maximized loss should be minimized, according to the paper
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_avg_loss  += loss
            success_rollout += 1

        total_avg_loss = torch.div(total_avg_loss, success_rollout)
        train_losses.append(total_avg_loss.item())
        print("Loss at epoch {} is {}".format(epoch+1, total_avg_loss[0]))
        print("Current Train Losses: ", train_losses)

        print("FINAL RESULT::" , train_losses)
        path = "./saved_models/model_maintrain" + ".tar"
        torch.save({'model_state_dict': model.state_dict() , 'emb_model_state_dict': emb_model.state_dict() }, path)
        print("Model Saved!!")
        
if __name__=='__main__':
    emb_model = Embedding(token_dim=15,value_dim=40, type=BitVector16)
    model = Network(emb_model.emb_dim,len(BitVectorLang.tokens))
    epochs = 2
    programs, IOs = DataLoader(10,10)
    PreTrain(emb_model, model, programs, IOs ,epochs)
    Train(emb_model, model, IOs, epochs)
    print("Training finished.")
