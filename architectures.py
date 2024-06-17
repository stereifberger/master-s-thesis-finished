from imports import *
#https://medium.com/@hugmanskj/hands-on-implementing-a-simple-mathematical-calculator-using-sequence-to-sequence-learning-85b742082c72
class Encoder_LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)  # Embedding layer
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)  # RNN layer

    def forward(self, src):
        embedded = self.embedding(src)  # Convert input tokens to embeddings
        output, (hidden, cell) = self.rnn(embedded)  # Forward pass through RNN

        # In LSTM, h_n and c_n store the hidden state and cell state of the last time step, respectively.
        # For more information, you can check the PyTorch documentation for LSTM
        # at: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        #print(f"HIDDEN SHAPE: {hidden.shape}")
        return hidden

# Decoder Definition
class Decoder_LSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)  # RNN layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, input):
        output, (hidden, cell) = self.rnn(input)  # Forward pass through RNN
        #print(f"OUTPUT:{output.shape}")
        prediction = self.fc_out(output)  # Predict next token
        #print(f"PREDICTION:{prediction.shape}")
        return prediction

######################FFN on basis of LSTM below by GPT
class Encoder_FFN(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)  # Embedding layer
        self.fc_hidden = nn.Linear(emb_dim, emb_dim)  # Feedforward layer

    def forward(self, src):
        embedded = self.embedding(src)  # Convert input tokens to embeddings
        hidden = self.fc_hidden(embedded.mean(dim=1))  # Pass embeddings through the feedforward layer
        return hidden.unsqueeze(0)  # Make hidden compatible with LSTM hidden/cell state shape

# Decoder Definition
class Decoder_FFN(nn.Module):
    def __init__(self, output_dim, emb_dim):
        super().__init__()
        self.fc_hidden = nn.Linear(emb_dim, emb_dim)  # Feedforward layer
        self.fc_out = nn.Linear(emb_dim, output_dim)  # Output layer

    def forward(self, input):
        hidden = self.fc_hidden(input.mean(dim=1))  # Forward pass through the feedforward layer
        prediction = self.fc_out(hidden)  # Predict next token
        return prediction

############################ RNN above rewritten by GPT
class Encoder_RNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)  # Embedding layer
        self.rnn = nn.RNN(emb_dim, hidden_dim, n_layers, batch_first=True)  # RNN layer
    
    def forward(self, src):
        embedded = self.embedding(src)  # Convert input tokens to embeddings
        output, hidden = self.rnn(embedded)  # Forward pass through RNN
        # Hidden state at the last time step for each layer
        # For more information, you can check the PyTorch documentation for RNN
        # at: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        return hidden

# Decoder using RNN
class Decoder_RNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers):
        super().__init__()
        self.output_dim = output_dim
        self.rnn = nn.RNN(emb_dim, hidden_dim, n_layers, batch_first=True)  # RNN layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)  # Output layer
    
    def forward(self, input):
        output, hidden = self.rnn(input)  # Forward pass through RNN
        prediction = self.fc_out(output)  # Predict next token
        return prediction


# Seq2Seq Model Integration
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg_len):
        trg_len = trg_len 
        hidden = self.encoder(src)  # Initial hidden state from encoder.

        # get top layer hidden states
        last_layer_hidden = hidden[-1] # [batch_size, dim]
        #print(f"LAST LAYER HIDDEN: {last_layer_hidden.shape}")

        # copy
        #print(last_layer_hidden.shape)

        dec_input = last_layer_hidden.unsqueeze(1).expand(-1, trg_len, -1)
        #print(f"DEC_INPUT: {dec_input.shape}")
        # dec_input : [batch_size, target_length, dim]
        output = self.decoder(dec_input)

        return output  # [batch_size, target_length, num_output_label]

############################## Transformer written on basis of LSTM above by GPT
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, hidden_dim, num_layers, dropout, max_seq_len=100):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        embedded = self.embedding(src) + self.positional_encoding[:, :src.size(1), :].to(src.device)
        embedded = embedded.permute(1, 0, 2)  # [seq_len, batch_size, emb_dim]
        output = self.transformer_encoder(embedded)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, num_heads, hidden_dim, num_layers, max_seq_len=100):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, emb_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_dim, output_dim)
    
    def forward(self, tgt, memory):
        embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        embedded = embedded.permute(1, 0, 2)  # [seq_len, batch_size, emb_dim]
        output = self.transformer_decoder(embedded, memory)
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, emb_dim]
        prediction = self.fc_out(output)
        return prediction

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg_len):
        memory = self.encoder(src)
        tgt = torch.zeros((src.size(0), trg_len), device=self.device, dtype=torch.long)
        output = self.decoder(tgt, memory)
        return output


"""
Title: PyTorch For Deep Learning — Feed Forward Neural Network
Author: Ashwin Prasad
Date: 11.09.2020
URL: https://medium.com/analytics-vidhya/pytorch-for-deep-learning-feed-forward-neural-network-d24f5870c18
"""
class ffn(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate, input_size_in):
        super(ffn, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(input_size_in, hidden_size)
        self.l3 = nn.Linear(input_size_in, hidden_size)
        self.l4 = nn.Linear(input_size_in, hidden_size)
        self.l5 = nn.Linear(input_size_in, hidden_size)
        self.l6 = nn.Linear(input_size_in, hidden_size)
        self.l7 = nn.Linear(input_size_in, hidden_size)
        self.l8 = nn.Linear(input_size_in, hidden_size)
        self.l9 = nn.Linear(input_size_in, hidden_size)
        self.relu = nn.ReLU()
        self.l10 = nn.Linear(hidden_size, output_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, len):
        output = self.l1(x) 
        output = self.l2(x) 
        output = self.relu(output)
        output = self.dropout1(output)
        output = self.l3(x) 
        output = self.l4(x) 
        output = self.l5(x) 
        output = self.l6(x) 
        output = self.l7(x) 
        output = self.l8(x) 
        output = self.l9(x) 
        output = self.relu(output)
        output = self.dropout2(output)
        output = self.l10(output)
        return output

import torch
import torch.nn as nn


"""
Description: RNN, LSTM and Transformer generated with feedforward network class as template
Generated by: GPT-4
Date: 2024-04-05
URL of Service: https://platform.openai.com/playground
"""
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, len):
        # x: (batch_size, length_input_data, input_size)
        h_0 = torch.zeros(1, x.size(0), self.rnn.hidden_size)  # Initial hidden state
        h_0 = h_0.to(torch.device('cuda'))
        print(h_0.device)
        out, _ = self.rnn(x, h_0)
        # out: (batch_size, length_input_data, hidden_size)
        
        # Apply the fully connected layer to each timestep
        out = self.fc(out)  # (batch_size, length_input_data, output_size)
        return out

class rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 10):
        super(rnn, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Linear layer that maps from hidden state space to output space
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, len):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        # only interested in the last time step output
        out = self.fc(out)
        return out

# LSTM
class lst(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=10):
        super(lst, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Linear layer that maps from hidden state space to output space
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, len):
        # Initialize hidden and cell states with zeros
        # LSTM has two states: hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out