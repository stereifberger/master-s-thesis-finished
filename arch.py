from libs import *

# https://medium.com/analytics-vidhya/pytorch-for-deep-learning-feed-forward-neural-network-d24f5870c18
#dataset
from torch.utils.data import Dataset, DataLoader

class dset(data):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.length

#dataset = diabetesdataset(x,y)
#importing the datasetfrom sklearn.datasets import load_diabetes
#data = load_diabetes()
#x = data['data']
#y = data['target']#shape
#print('shape of x is : ',x.shape)
#print('shape of y is : ',y.shape)

#class diabetesdataset(Dataset):
 #   def __init__(self,x,y):
 #       self.x = torch.tensor(x,dtype=torch.float32)
 #       self.y = torch.tensor(y,dtype=torch.float32)
 #       self.length = self.x.shape[0]  def __getitem__(self,idx):
 #       return self.x[idx],self.y[idx]  def __len__(self):
 #       return self.length
#dataset = diabetesdataset(x,y)

#dataloader
dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=100)

#creeating the networkfrom torch import nnclass net(nn.Module):
  def __init__(self,input_size,output_size):
  super(net,self).__init__()
    self.l1 = nn.Linear(input_size,5)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(5,output_size)  def forward(self,x):
    output = self.l1(x) 
    output = self.relu(output)
    output = self.l2(output)
    return output

model = net(x.shape[1],1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
epochs = 1500

# training
costval = []for j in range(epochs):
  for i,(x_train,y_train) in enumerate(dataloader):    #prediction
    y_pred = model(x_train)
    
    #calculating loss
    cost = criterion(y_pred,y_train.reshape(-1,1))
  
    #backprop
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
  if j%50 == 0:
    print(cost)
    costval.append(cost)