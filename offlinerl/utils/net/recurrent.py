import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentGRU(nn.Module):
    def __init__(self, input_dim, device=None, rnn_hidden_dim=128, rnn_layer_num=1):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layer_num = rnn_layer_num
        self.device = device
        
        self.GRU = nn.GRU(
            input_dim, 
            rnn_hidden_dim, 
            rnn_layer_num, 
            batch_first=True
        )
        
    def forward(self, x, lens, pre_hidden=None, sequential=False):
        if pre_hidden is None:
            pre_hidden = self.zero_hidden(batch_size=x.shape[0])
        if len(pre_hidden.shape) == 2:
            pre_hidden = torch.unsqueeze(pre_hidden, dim=0)
            
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        
        output, hidden = self.GRU(packed, pre_hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
    
    def zero_hidden(self, batch_size):
        return torch.zeros([self.rnn_layer_num, batch_size, self.rnn_hidden_dim]).to(self.device)
        
    def get_hidden(self, obs, last_actions, lens):
        pre_hidden = self.zero_hidden(len(lens))
        x = torch.cat([obs, last_actions], dim=-1)
        output, hidden = self(x, lens, pre_hidden=pre_hidden)
        return output


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, device=None, rnn_hidden_dim=128, rnn_layer_num=1):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layer_num = rnn_layer_num
        self.device = device

        self.input_embed = nn.Linear(self.input_dim, self.rnn_hidden_dim)

        self.trans_enc_layer = nn.TransformerEncoderLayer(d_model=self.rnn_hidden_dim, nhead=1, batch_first=True)
        self.trans_enc = nn.TransformerEncoder(self.trans_enc_layer, num_layers=self.rnn_layer_num)

        self.output_layer = nn.Linear(self.rnn_hidden_dim, output_dim)
        self.inputs = []
        self.max_size = 20

    def forward(self, x, sequential=False):
        if not sequential:
            self.inputs.append(x)
            if len(self.inputs) > self.max_size:
                self.inputs.pop(0)
            if len(x.shape) == 2:
                inputs = torch.stack(self.inputs, dim=1)
            else:
                inputs = torch.cat(self.inputs, dim=1)
        else:
            inputs = x

        inputs_embed = self.input_embed(inputs)
        masks = nn.Transformer.generate_square_subsequent_mask(inputs.shape[1]).to(inputs.device)
        outputs = self.trans_enc(inputs_embed, masks)

        return self.output_layer(outputs)

    def zero_hidden(self, batch_size):
        self.inputs = []
        return torch.zeros([self.rnn_layer_num, batch_size, self.rnn_hidden_dim]).to(self.device)

    def get_hidden(self, obs, last_actions, lens):
        pre_hidden = self.zero_hidden(len(lens))
        x = torch.cat([obs, last_actions], dim=-1)
        inputs = x
        inputs_embed = self.input_embed(inputs)
        masks = nn.Transformer.generate_square_subsequent_mask(inputs.shape[1]).to(inputs.device)
        outputs = self.trans_enc(inputs_embed, mask=masks)
        output = outputs
        if self.with_tanh:
            output = F.tanh(output)
        return output


class TransformerEnc(nn.Module):
    def __init__(self, input_dim, device=None, rnn_hidden_dim=128, rnn_layer_num=1, with_tanh=True, max_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layer_num = rnn_layer_num
        self.device = device

        self.input_embed = nn.Linear(self.input_dim, self.rnn_hidden_dim)

        self.trans_enc_layer = nn.TransformerEncoderLayer(d_model=self.rnn_hidden_dim, nhead=1, batch_first=True)
        self.trans_enc = nn.TransformerEncoder(self.trans_enc_layer, num_layers=self.rnn_layer_num)

        # self.output_embed = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.with_tanh = with_tanh

        self.inputs = []
        self.max_size = max_size
        self.pop_pos = 0

    def forward(self, x, lens, pre_hidden=None, sequential=False):
        if not sequential:
            self.inputs.append(x)
            if len(self.inputs) > self.max_size:
                self.inputs.pop(self.pop_pos)
            if len(x.shape) == 2:
                inputs = torch.stack(self.inputs, dim=1)
            else:
                inputs = torch.cat(self.inputs, dim=1)
        else:
            inputs = x

        inputs_embed = self.input_embed(inputs)
        masks = nn.Transformer.generate_square_subsequent_mask(inputs.shape[1]).to(inputs.device)
        outputs = self.trans_enc(inputs_embed)
        # outputs = self.output_embed(outputs_embed)

        if not sequential:
            output = outputs[:, -1:]
            hidden = outputs[:, -1:]
        else:
            output = outputs
            hidden = outputs
        if self.with_tanh:
            output = F.tanh(output)
            hidden = F.tanh(hidden)

        return output, hidden

    def zero_hidden(self, batch_size):
        self.inputs = []
        return torch.zeros([self.rnn_layer_num, batch_size, self.rnn_hidden_dim]).to(self.device)

    def get_hidden(self, obs, last_actions, lens):
        pre_hidden = self.zero_hidden(len(lens))
        x = torch.cat([obs, last_actions], dim=-1)
        inputs = x
        inputs_embed = self.input_embed(inputs)
        masks = nn.Transformer.generate_square_subsequent_mask(inputs.shape[1]).to(inputs.device)
        outputs = self.trans_enc(inputs_embed, mask=masks)
        output = outputs
        if self.with_tanh:
            output = F.tanh(output)
        return output