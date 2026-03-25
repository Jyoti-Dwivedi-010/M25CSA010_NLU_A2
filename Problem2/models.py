import torch
import torch.nn as nn
import math

# vanilla RNN cell and module implemented from scratch without using PyTorch's built-in RNN layers, allowing for a deeper understanding of the underlying mechanics of recurrent neural networks. The VanillaRNNCell class defines the computations for a single time step, while the VanillaRNNModule class manages the sequence processing and output generation for an entire input sequence.
class VanillaRNNCell(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(VanillaRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        # x: (batch_size, input_size)
        # hidden: (batch_size, hidden_size)
        new_hidden = self.tanh(self.i2h(x) + self.h2h(hidden))
        return new_hidden
    

# module for vanilla RNN that processes an entire sequence of inputs, utilizing the VanillaRNNCell for each time step, and produces output predictions for each time step based on the hidden states. The module also includes a method to initialize the hidden state and a fully connected layer to map the hidden state to the output vocabulary size.
class VanillaRNNModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn_cells = nn.ModuleList([VanillaRNNCell(hidden_size, hidden_size) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        batch_size, seq_len = x.size()
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
            
        embedded = self.embed(x)
        out_seq = []
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            for layer in range(self.n_layers):
                x_t = self.rnn_cells[layer](x_t, hidden[layer])
                hidden[layer] = x_t
            out_seq.append(self.fc_out(x_t).unsqueeze(1))
            
        output = torch.cat(out_seq, dim=1) # (batch, seq, vocab)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.n_layers)]

# lstm cell and bidirectional lstm module implemented from scratch, where the LSTM cell (LSTMCellScratch) defines the computations for the input gate, forget gate, cell candidate, and output gate, while the BLSTMModule manages the forward and backward passes through the sequence, combining the outputs from both directions to produce final predictions. The module also includes a method to count the number of trainable parameters in the model.
class LSTMCellScratch(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Computes i, f, g, o all in one linear layer
        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)
        
    def forward(self, x, state):
        hx, cx = state
        
        gates = self.ih(x) + self.hh(hx)
        # Split gates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

class BLSTMModule(nn.Module):
    
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # Forward cell and backward cell
        self.cell_f = LSTMCellScratch(hidden_size, hidden_size)
        self.cell_b = LSTMCellScratch(hidden_size, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x, hidden=None):
        batch, seq_len = x.size()
        
        if hidden is None:
            # Tuple for (fwd_h, fwd_c, bck_h, bck_c)
            device = x.device
            hidden = (torch.zeros(batch, self.hidden_size).to(device),
                      torch.zeros(batch, self.hidden_size).to(device),
                      torch.zeros(batch, self.hidden_size).to(device),
                      torch.zeros(batch, self.hidden_size).to(device))
            
        hf, cf, hb, cb = hidden
        embedded = self.embed(x)
        
        fwd_states = []
        bck_states = []
        
        # Forward computation
        for t in range(seq_len):
            hf, cf = self.cell_f(embedded[:, t, :], (hf, cf))
            fwd_states.append(hf)
            
        # Backward computation
        for t in range(seq_len - 1, -1, -1):
            hb, cb = self.cell_b(embedded[:, t, :], (hb, cb))
            bck_states.insert(0, hb)
            
        out_seq = []
        for t in range(seq_len):
            combined = torch.cat((fwd_states[t], bck_states[t]), dim=1)
            out_seq.append(self.fc_out(combined).unsqueeze(1))
            
        return torch.cat(out_seq, dim=1), (hf, cf, hb, cb)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# rnn attention module that implements a vanilla RNN with an attention mechanism, allowing the model to focus on different parts of the input sequence when generating each output. The RNNAttention class defines the architecture and forward pass for this model, including the attention score calculation and context vector computation, which are used to enhance the hidden state before producing the final output predictions.
class RNNAttention(nn.Module):
    
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.cell = VanillaRNNCell(hidden_size, hidden_size)
        
        self.attn_W = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        batch, seq_len = x.size()
        device = x.device
        if hidden is None:
            hidden = torch.zeros(batch, self.hidden_size).to(device)
            
        embedded = self.embed(x)
        outputs = []
        encoder_states = [] # To store states to attend to
        
        # Standard RNN Forward
        for t in range(seq_len):
            hidden = self.cell(embedded[:, t, :], hidden)
            encoder_states.append(hidden.unsqueeze(1))
            
        enc_states = torch.cat(encoder_states, dim=1) # (batch, seq, hidden)
        
        # Apply attention over history for each step's output
        for t in range(seq_len):
            # Target hidden state
            ht = enc_states[:, t, :]
            # Compute limited attention up to t
            if t == 0:
                context = ht
            else:
                history = enc_states[:, :t, :]
                ht_expanded = ht.unsqueeze(1).repeat(1, t, 1)
                
                # Attention score calculation v^T tanh(W [h_t; h_{history}])
                energy = torch.tanh(self.attn_W(torch.cat((ht_expanded, history), dim=2)))
                attn_weights = torch.softmax(self.attn_v(energy).squeeze(2), dim=1) # (batch, t)
                
                context = torch.bmm(attn_weights.unsqueeze(1), history).squeeze(1)
                
            out = self.fc_out(context + ht)
            outputs.append(out.unsqueeze(1))
            
        return torch.cat(outputs, dim=1), enc_states[:, -1, :]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
