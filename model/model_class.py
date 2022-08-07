import torch
import torch.nn as nn
import torch.nn.functional as F




def get_model() -> nn.Module:

    class Model(nn.Module):

        def __init__(self):
            super().__init__()

            self.input = nn.Linear(10, 32)

            self.fc1 = nn.Linear(32, 128)
            self.bnorm1 = nn.BatchNorm1d(128)

            self.fc2 = nn.Linear(128, 512)
            self.bnorm2 = nn.BatchNorm1d(512)

            self.fc3 = nn.Linear(512, 64)
            self.bnorm3 = nn.BatchNorm1d(64)

            self.fc4 = nn.Linear(64, 32)
            self.bnorm4 = nn.BatchNorm1d(32)

            self.output = nn.Linear(32, 8)

        def forward(self, x):
            # print(f'training={self.training}')

            x = F.relu(self.input(x))

            x = F.relu(self.bnorm1(self.fc1(x)))
            x = F.dropout(x, p=0.25, training=self.training)

            x = F.relu(self.bnorm2(self.fc2(x)))
            x = F.dropout(x, p=0.4, training=self.training)

            x = F.relu(self.bnorm3(self.fc3(x)))
            x = F.dropout(x, p=0.1, training=self.training)

            x = F.relu(self.bnorm4(self.fc4(x)))
            x = F.dropout(x, p=0.05, training=self.training)

            x = F.relu(self.output(x))
            # x = self.output(x)

            return x


    return Model()




