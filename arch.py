from libs import *

# https://medium.com/analytics-vidhya/pytorch-for-deep-learning-feed-forward-neural-network-d24f5870c18
#dataset

class dset(data):
    def __init__(self,x,y):
        self.length = self.x.shape[0]
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
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