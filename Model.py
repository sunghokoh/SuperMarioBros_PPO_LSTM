import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical

class ActorCriticNet(nn.Module):
    def __init__(self, N_action):
        super(ActorCriticNet, self).__init__()

        # Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.ConvOutSize = self.get_conv_out_size()

        self.Lstm = nn.LSTMCell(self.ConvOutSize * self.ConvOutSize * 64, 512)

        self.Pi = nn.Linear(512, N_action)
        self.V = nn.Linear(512, 1)

        self.initialize_weights()
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    
    def get_conv_out_size(self):
        test_tensor = torch.FloatTensor(1, 1, 84, 84)
        out_tensor = self.conv4(self.conv3(self.conv2(self.conv1(test_tensor))))
        conv_out_size = out_tensor.size()[-1]
        return conv_out_size
    
    def forward(self, x, hidden):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, self.ConvOutSize * self.ConvOutSize * 64)

        h, c = self.Lstm(x, hidden)

        prob = self.Pi(h)
        prob = F.softmax(prob, dim=-1)
        
        value = self.V(h)

        return prob, value, (h, c)

class ActorCriticNet_MA(nn.Module):
    def __init__(self):
        super(ActorCriticNet_MA, self).__init__()

        # Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.ConvOutSize = self.get_conv_out_size()

        self.Lstm = nn.LSTMCell(self.ConvOutSize * self.ConvOutSize * 64, 512)

        self.Pi = nn.Linear(512, 4) # right, left, A, B
        self.V = nn.Linear(512, 1)

        self.initialize_weights()
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def get_conv_out_size(self):
        test_tensor = torch.FloatTensor(1, 1, 84, 84)
        out_tensor = self.conv4(self.conv3(self.conv2(self.conv1(test_tensor))))
        conv_out_size = out_tensor.size()[-1]
        return conv_out_size
    
    def forward(self, x, hidden):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, self.ConvOutSize * self.ConvOutSize * 64)

        h, c = self.Lstm(x, hidden)

        prob = self.Pi(h)
        prob = torch.sigmoid(prob)
        
        value = self.V(h)

        return prob, value, (h, c)



