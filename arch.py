from libs import *
from torch import nn

# https://medium.com/analytics-vidhya/pytorch-for-deep-learning-feed-forward-neural-network-d24f5870c18
# https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/10
#dataset


#class dset(Dataset):
#    def __init__(self,x):
#        self.x = x
#        self.length = self.x.shape[0]
#    def __getitem__(self, idx):
#        return self.x[idx], idx
#    def __len__(self):
#        return self.length


class dset(Dataset):
    def __init__(self,x):
        self.length = self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx]
    def __len__(self):
        return self.length

#creeating the networkfrom torch import nnclass net(nn.Module): 
class net(nn.Module):
    def __init__(self,input_size,output_size):
        super(net,self).__init__()
        self.l1 = nn.Linear(input_size,5)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(5,output_size)

    def forward(self,x):
        output = self.l1(x) 
        output = self.relu(output)
        output = self.l2(output)
        return output
    