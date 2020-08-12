import torch
from synthrl.agent.rl import Network
from synthrl.agent.train import RnnInit


from synthrl.language.bitvector.embedding import Embedding
from synthrl.value.bitvector import BitVector
from synthrl.value.bitvector import BitVector16
from synthrl.language.bitvector.lang import BitVectorLang

pgm = BitVectorLang()
pgm = BitVectorLang()
pgm.production("bop")
pgm.production("+")


io_inputs = [  (BitVector16(5),BitVector16(10))    ,
            (BitVector16(8),BitVector16(2) )    ,
            (BitVector16(7),BitVector16(20))    ,
            (BitVector16(5),BitVector16(9) )    ] 

io_outputs = [ BitVector16(15), 
            BitVector16(10), 
            BitVector16(27), 
            BitVector16(14) ]






emb_model = Embedding(token_dim=15,value_dim=40, type=BitVector16)
model = Network(emb_model.emb_dim,len(BitVectorLang.tokens))
hidden, outputs = RnnInit(10,1,4)

for token in pgm.tokenize():
    emb_val = emb_model( [token], [io_inputs], [io_outputs])
    y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
    query = query.permute(1,0,2)
    outputs = torch.cat((outputs,query),dim=1)
print(y_policy)
####After Model Deriviation####

chk = torch.load("./saved_models/model_maintrain.tar")
model.load_state_dict(chk['model_state_dict'])
emb_model.load_state_dict(chk['emb_model_state_dict']) 


hidden, outputs = RnnInit(10,1,4)

for token in pgm.tokenize():
    emb_val = emb_model( [token], [io_inputs], [io_outputs])
    y_policy, y_value, query, hidden = model(emb_val,hidden,outputs)
    query = query.permute(1,0,2)
    outputs = torch.cat((outputs,query),dim=1)
print(y_policy)
print(sorted(BitVectorLang.tokens))