import torch
import torch.nn as nn
import torch.nn.functional as F




def get_model() -> nn.Module:


    class Model(nn.Module):

        def __init__(self):

            super().__init__()

            self.input = nn.Linear(10, 64)

            self.fc1 = nn.Linear(64, 128)
            self.bnorm1 = nn.BatchNorm1d(128)

            self.fc2 = nn.Linear(128, 512)
            self.bnorm2 = nn.BatchNorm1d(512)

            self.fc3 = nn.Linear(512, 1024)
            self.bnorm3 = nn.BatchNorm1d(1024)

            self.fc4 = nn.Linear(1024, 512)
            self.bnorm4 = nn.BatchNorm1d(512)

            self.fc5 = nn.Linear(512, 128)
            self.bnorm5 = nn.BatchNorm1d(128)

            self.fc6 = nn.Linear(128, 64)
            self.bnorm6 = nn.BatchNorm1d(64)

            self.output = nn.Linear(64, 8)


        def forward(self, x):

            # print(f'training={self.training}')

            x = F.relu(self.input(x))  # 64

            x = F.relu(self.bnorm1(self.fc1(x)))  # 128
            x = F.dropout(x, p=0.07, training=self.training)

            x = F.relu(self.bnorm2(self.fc2(x)))  # 512
            x = F.dropout(x, p=0.12, training=self.training)

            x = F.relu(self.bnorm3(self.fc3(x)))  # 1024
            x = F.dropout(x, p=0.25, training=self.training)

            x = F.relu(self.bnorm4(self.fc4(x)))  # 512
            x = F.dropout(x, p=0.12, training=self.training)

            x = F.relu(self.bnorm5(self.fc5(x)))  # 128
            x = F.dropout(x, p=0.07, training=self.training)

            x = F.relu(self.bnorm6(self.fc6(x)))  # 64
            # x = F.dropout(x, p=0.05, training=self.training)

            x = F.relu(self.output(x))  # 8
            # x = self.output(x)

            return x



    return Model()





