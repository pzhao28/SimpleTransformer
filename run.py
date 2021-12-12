import time
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
from transformerModel import make_model
from dataGen import data_gen
from utils import *

def run_epoch(data_iter, model, loss_compute):
  #Standard Training and Logging Function
  start = time.time()
  total_tokens = 0
  total_loss = 0
  tokens = 0
  for i, batch in enumerate(data_iter):
    out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
    loss = loss_compute(out, batch.trg_y, batch.ntokens)
    total_loss += loss
    total_tokens += batch.ntokens
    tokens += batch.ntokens
    if i % 50 == 1:
      elapsed = time.time() - start
      print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %(i, loss / batch.ntokens, tokens/elapsed))
      start = time.time()
      tokens = 0
  return total_loss/total_tokens

#Train the simple copy task
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V,V,N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))
for epoch in range(10):
  model.train()
  run_epoch(data_gen(V,30,20), model, SimpleLossCompute(model.generator, criterion, model_opt))
  model.eval()
  print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))

model.eval()
src = torch.LongTensor([[1,3,2,4,5,6,7,8,9,10]])
src_mask = torch.ones(1,1,10)
print(greedy_decode(model,src,src_mask,max_len=10,start_symbol=1))