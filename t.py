import torch
import torch.nn as nn

def log_softmax(x): return x - x.exp().sum(-1,keepdim=True).log()

def logsumexp(x):
    m = x.max(-1)[0]
    return m + (x-m[:,None]).exp().sum(-1).log()

def log_softmax(x): return x - x.logsumexp(-1,keepdim=True)

def nll(input, target): return -input[range(target.shape[0]), target].mean()

targets = torch.tensor([3,6,2,8,1])
preds = (torch.arange(50).reshape(5,10) * 0.01).exp() * 0.1

sm_pred = log_softmax(preds)
l1 = nll(sm_pred, targets)

loss_fn = nn.CrossEntropyLoss()
l2 = loss_fn(preds, targets)
print(l1, l2)