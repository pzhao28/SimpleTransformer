import numpy as np
import torch
import seaborn
seaborn.set_context(context="talk")
from transformerModel import subsequent_mask

class Batch:
  #Object for holding a batch of data with mask during training
  def __init__(self, src, trg=None, pad=0):
    self.src = src
    self.src_mask = (src!=pad).unsqueeze(-2)
    if trg is not None:
      self.trg = trg[:, :-1]
      self.trg_y = trg[:,1:] #debug trg[:,:-1]
      self.trg_mask = self.make_std_mask(self.trg, pad)
      self.ntokens = (self.trg_y!=pad).sum()

  @staticmethod
  def make_std_mask(tgt, pad):
    #Create a mask to hide padding and future words
    tgt_mask = (tgt!=pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask

def data_gen(V, batch, nbatches):
  #Generate rnadom data for a src-tgt cipy task
  for i in range(nbatches):
    data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
    data[:,0]=1
    src = data.requires_grad_(False)
    tgt = data.requires_grad_(False)
    yield Batch(src, tgt, 0)
