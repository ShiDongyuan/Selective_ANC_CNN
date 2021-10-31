import numpy as np 
import torch
from torch.autograd.grad_mode import F

embedding_size = 4 
num_classes    = 2

Wc = torch.nn.Parameter(torch.Tensor(embedding_size, num_classes)).cpu()
Wn = Wc.detach().numpy()

print(Wn.dtype)

import argparse

parser = argparse.ArgumentParser(description='PyTorch L-Softmax MNIST Example')
parser.add_argument('--margin', type=int, default=4, metavar='M',
                        help='the margin for the l-softmax formula (m=1, 2, 3, 4)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 0.0005)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--vis', default=False, metavar='V',
                        help='enables visualizing 2d features (default: False).')
args = parser.parse_args()
print(args)

a = torch.arange(16000*6)
print(a.shape[0])

assert  a.shape[0] % 16000 == 0, 'The length of the primary noise is not an integral multiple of fs'

i = 1
 
c = torch.rand(2, 3)
print(c.shape)
d = c.unsqueeze(1)
print(d.shape)

print(f'The tensors dimension is {d.dim()}')

x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).unsqueeze(0)
print(x)
print(x.shape)
y = torch.roll(x,1,1)

print(y)