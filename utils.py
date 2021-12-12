import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import seaborn
seaborn.set_context(context="talk")
from transformerModel import subsequent_mask



class NoamOpt:
  #Optim wrapper that implements rate
  def __init__(self, model_size, factor, warmup, optimizer):
    self.optimizer = optimizer
    self._step = 0
    self.warmup = warmup
    self.factor = factor
    self.model_size = model_size
    self._rate = 0

  def step(self):
    #Update parameters and rate
    self._step += 1
    rate = self.rate()
    for p in self.optimizer.param_groups:
      p['lr'] = rate
    self._rate = rate
    self.optimizer.step()

  def rate(self, step = None):
    #Implement 'lrate' above
    if step is None:
      step = self._step
    return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
  return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.opim.Adam(model.parameters(),lr=0, betas=(0.9,0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
  #Label smoothing is a regularization technique that addresses both problems of overfitting and overconfidence in classification.
  #Implement label smoothing
  def __init__(self, size, padding_idx, smoothing=0.0):
    super().__init__()
    self.criterion = nn.KLDivLoss(reduction='sum')
    self.padding_idx = padding_idx
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
    self.size = size
    self.true_dist = None

  def forward(self, x, target):
    assert x.size(1) == self.size
    true_dist = x.clone()
    true_dist.fill_(self.smoothing / (self.size-2))
    true_dist.scatter_(1,target.unsqueeze(1), self.confidence)
    true_dist[:, self.padding_idx] = 0
    mask = torch.nonzero(target == self.padding_idx)
    if mask.dim()>0:
      true_dist.index_fill_(0, mask.squeeze(), 0.0)
    self.true_dist = true_dist
    return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
  #A simple loss compute and train function
  def __init__(self, generator, criterion, opt=None):
    self.generator = generator
    self.criterion = criterion
    self.opt = opt

  def __call__(self, x, y, norm):
    x = self.generator(x)
    loss = self.criterion(x.contiguous().view(-1,x.size(-1)), y.contiguous().view(-1))/norm
    loss.backward()
    if self.opt is not None:
      self.opt.step()
      self.opt.optimizer.zero_grad()
    return loss.item()*norm

def greedy_decode(model, src, src_mask, max_len, start_symbol):
  memory = model.encode(src, src_mask)
  ys = torch.ones(1,1).fill_(start_symbol).type_as(src)
  for i in range(max_len-1):
    out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src))
    prob = model.generator(out[:,-1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word[0]
    ys = torch.cat([ys, torch.ones(1,1).type_as(src).fill_(next_word)],dim=1)
  return ys