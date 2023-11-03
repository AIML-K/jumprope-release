import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

raw_data = pd.read_csv('input_example.csv').to_numpy()

type_column = np.full((raw_data.shape[0], 1), 'others')
logit_column = np.zeros((raw_data.shape[0], 1), dtype=int)

raw_data = np.hstack((raw_data, type_column, logit_column))

sensor_data = np.array(raw_data[:,0], dtype= int)
time_data = np.array(raw_data[:,1], dtype= float)
type_data = raw_data[:,2]


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.leakyrelu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# for 92_mlp_model.pt
#
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(4, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 64)
#         self.fc4 = nn.Linear(64, 1)
#         self.leakyrelu = nn.LeakyReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(p=0.3)
#         self.bn = nn.BatchNorm1d(64)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.dropout(x)
#         x = self.leakyrelu(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         x = self.leakyrelu(x)
#         x = self.fc3(x)
#         x = self.dropout(x)
#         x = self.leakyrelu(x)
#         x = self.fc4(x)
#         x = self.sigmoid(x)
#         return x


model = torch.load("mlp_model.pt")

for i in range(len(sensor_data) - 5):
    if sensor_data[i] == 1 and sensor_data[i+1] == 2 and sensor_data[i+2] == 3 and sensor_data[i+3] == 4 :
        with torch.no_grad():
            model.eval()
            inputs = torch.Tensor(time_data[i:i+4]).float()
            outputs = model(inputs)
        if outputs >= 0.5:
            type_data[i] = type_data[i+1] = type_data[i+2] = type_data[i+3] = 'double'
            raw_data[i,3] = raw_data[i+1,3] = raw_data[i+2,3] = raw_data[i+3,3] = round(float(outputs),2)
        else:
            type_data[i] = type_data[i+1] = type_data[i+2] = type_data[i+3] = 'single'
            raw_data[i,3] = raw_data[i+1,3] = raw_data[i+2,3] = raw_data[i+3,3] = round(1 - float(outputs),2)
    elif sensor_data[i] == 1 and sensor_data[i+1] == 2 and sensor_data[i+2] == 2 and sensor_data[i+3] == 3 and sensor_data[i+4] == 4:
        with torch.no_grad():
            model.eval()
            inputs = torch.Tensor(np.delete(time_data[i:i+5], 2)).float()
            outputs = model(inputs)
        if outputs >= 0.5:
            type_data[i] = type_data[i+1] = type_data[i+3] = type_data[i+4] = 'double'
            raw_data[i,3] = raw_data[i+1,3] = raw_data[i+3,3] = raw_data[i+4,3] = round(float(outputs),2)
        else:
            type_data[i] = type_data[i+1] = type_data[i+3] = type_data[i+4] = 'single'
            raw_data[i,3] = raw_data[i+1,3] = raw_data[i+3,3] = raw_data[i+3,3] = round(1 - float(outputs),2)       
    elif sensor_data[i] == 1 and sensor_data[i+1] == 2 and sensor_data[i+2] == 3 and sensor_data[i+3] == 3 and sensor_data[i+4] == 4:
        with torch.no_grad():
            model.eval()
            inputs = torch.Tensor(np.delete(time_data[i:i+5], 3)).float()
            outputs = model(inputs)
        if outputs >= 0.5:
            type_data[i] = type_data[i+1] = type_data[i+2] = type_data[i+4] = 'double'
            raw_data[i,3] = raw_data[i+1,3] = raw_data[i+2,3] = raw_data[i+4,3] = round(float(outputs),2)
        else:
            type_data[i] = type_data[i+1] = type_data[i+2] = type_data[i+4] = 'single'
            raw_data[i,3] = raw_data[i+1,3] = raw_data[i+2,3] = raw_data[i+4,3] = round(1 - float(outputs),2)       

raw_data[:,2] = type_data
raw_data = np.insert(raw_data, 0,['sensor','time','type','logit'], axis = 0)

np.savetxt("output_example.csv", raw_data, fmt='%s', delimiter= ',')
