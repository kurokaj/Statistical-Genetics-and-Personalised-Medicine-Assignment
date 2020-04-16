# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
 The linear regression module. 
Input (max) 40 and ouput 40 (DO NOT INCLUDE ID!)
"""
class linreg(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(linreg,self).__init__()
        self.lin = nn.Linear(inputsize, outputsize)
    
    def forward(self, x):
        y = self.lin(x)
        return y
    

"""
Train the model
Data is input one line at a time
Should we use batches? 
"""
def train_model(model, X_train, Y_train):
    # Choose the device for model and variables
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(X_train).cuda())
        labels = Variable(torch.from_numpy(Y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(X_train))
        labels = Variable(torch.from_numpy(Y_train))
    
    # Determine the criterions and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    losses = []
    
    epochs = 100
    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item() / X_train.shape[0])
        
        if epoch % 10 == 0:
            print("The loss is: " + str(loss))
         
    plt.plot(losses)
    print("Training done. Final loss: " + str(loss))       
                
"""
Do one prediction to see the form 
"""     
def test_model(model, X):
    x = torch.FloatTensor(X[0])
    out = model(x)
    print('input: %.4f, output: %.4f' % (x.item(), out.item()) )        
    
    
"""
Save the created linear model
"""
def save_model(model):
    torch.save(model.state_dict(), "E:/AALTO/Kevät2020/STATISTICAL GENETICS  & PERS. MED/Project/Statistical-Genetics-and-Personalised-Medicine-Assignment/Models")
    print("Model saved...")
    
    
def generate_pred(model, X):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
