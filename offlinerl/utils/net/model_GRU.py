import torch.nn
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from offlinerl.utils.net.common import miniblock

class Recurrent(nn.Module):
    def __init__(self, obs_dim, action_dim, device=None, rnn_hidden_size=128, rnn_layers=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_di9m = action_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.device = device
        
        self.GRU = nn.GRU(
            obs_dim + action_dim, 
            rnn_hidden_size, 
            rnn_layers, 
            batch_first=True
        )
        
    def forward(self, obs, last_action, lens, pre_hidden=None):
        if pre_hidden is None:
            pre_hidden = self.zero_hidden(batch_size=obs.shape[0])
            
        state_action_pair = torch.cat([obs, last_action], dim=-1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(state_action_pair, lens, batch_first=True, enforce_sorted=False)
        
        output, hidden = self.GRU(packed, pre_hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
    
    def zero_hidden(self, batch_size):
        return torch.zeros([self.rnn_layers, batch_size, self.rnn_hidden_size]).to(self.device)


class GRU_Model(nn.Module):
    def __init__(self, obs_dim, action_dim,device=None, lstm_hidden_units=128):
        super(GRU_Model, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.lstm_hidden_units = lstm_hidden_units
        self.GRU = nn.GRU(self.obs_dim + self.action_dim, lstm_hidden_units, batch_first=True)
    def forward(self, obs, last_acts, pre_hidden, lens):
        sta_acs = torch.cat([obs, last_acts], dim=-1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(sta_acs,lens,batch_first=True, enforce_sorted=False)
        if len(pre_hidden.shape) == 2:
            pre_hidden = torch.unsqueeze(pre_hidden, dim=0)
        output,_ = self.GRU(packed, pre_hidden)
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output
    def get_hidden(self, obs, last_actions, lens):
        pre_hidden = torch.zeros((1,len(lens),self.lstm_hidden_units)).to(self.device)
        return self(obs, last_actions, pre_hidden,lens)





