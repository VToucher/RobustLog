import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLstm(nn.Module):
    def __init__(self, units, vocab_size, embedding_dim):
        super(BiLstm, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, units, bidirectional=True, batch_first=True)
        
    def forward(self, inputs, mask):
        emb = self.embedding_layer(inputs)
        # import pdb; pdb.set_trace()
        lengths = mask.sum(dim=1).tolist()
        packed_emb = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm_layer(packed_emb)  # LSTM returns a tuple, we ignore the hidden state for now
        
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output

class Attention(nn.Module):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.units = units
        self.w = nn.Linear(units * 2, units)  # *2 because of bidirectional
        self.v = nn.Linear(units, 1)
        
    def forward(self, bilstm_ops, attention_mask=None):
        logits = self.w(bilstm_ops)  # [64, 64, 64]
        score = self.v(torch.tanh(logits)).squeeze(-1)  # [64, 64]
        
        if attention_mask is not None:
            score = score.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(score, dim=1)  # [64, 64]
        # import pdb; pdb.set_trace()
        context_vec = bilstm_ops * attention_weights.unsqueeze(-1)  # [64, 64, 64*2]
        context_vec = torch.sum(context_vec, dim=1)  # sentence level reprsentation [64, 128]
        
        return context_vec, attention_weights

class AttentionClassification(nn.Module):
    def __init__(self, num_class, units_, units, vocab_size, embedding_dim, mode=0):
        super(AttentionClassification, self).__init__()
        self.units_ = units_
        self.units = units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bilstm = BiLstm(units, vocab_size, embedding_dim)
        self.attention = Attention(units_)
        self.w = nn.Linear(units_ * 2, num_class)
        self.mode = mode
        if self.mode == 0:
            print("Enable Attention Classifier... 🚀")
        elif self.mode == 1:
            print("Enable Normal Classifier... 🔥")
        else:
            print("Not have such mode... 🤷‍♂️")
        
    def forward(self, input_x, mask):
        h = self.bilstm(input_x, mask)
        context_vec, attention_weights = self.attention(h, mask)
        
        if self.mode == 0:
            logits = self.w(context_vec)  # [64, 128] -> [64, num_class]
        elif self.mode == 1:
            logits = torch.mean(self.w(h), dim=1)
        else:
            return None, None
        
        return logits
