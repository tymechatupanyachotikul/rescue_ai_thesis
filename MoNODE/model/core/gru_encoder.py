import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pack_padded_sequence

class GRUEncoder(nn.Module):
    def __init__(self, output_dim, input_dim, rnn_output_size=20, H=50, act='relu'):
        super(GRUEncoder, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.rnn_output_size = rnn_output_size # number of outputs per output_dim
        self.rnn_hidden_to_latent = nn.Sequential(nn.Linear(self.rnn_output_size, H), nn.ReLU(True), nn.Linear(H, output_dim))
        self.gru = GRU(self.input_dim, self.rnn_output_size)

    def forward(self, data, lengths=None, run_backwards=True):
        if lengths is not None:
            lengths = lengths.clamp(min=1)
            N, T, D = data.shape 

            if run_backwards:
                T_idx = torch.arange(T, device=data.device).unsqueeze(0).expand(N, T)

                rev_idx = lengths.unsqueeze(1) - 1 - T_idx
                rev_idx = torch.clamp(rev_idx, min=0)

                data = torch.gather(data, 1, rev_idx.unsqueeze(-1).expand(-1, -1, D))

                mask = T_idx < lengths.unsqueeze(1)
                data = data * mask.unsqueeze(-1)

            packed_data = pack_padded_sequence(
                data, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed_data)
            final_hidden = hidden[-1]
        else:
            data = data.permute(1,0,2)  # (N, T, D) -> (T, N, D)
            if run_backwards:
                data = torch.flip(data, [0])  # (T, N, D)

            all_hidden, _ = self.gru(data) 
            final_hidden = all_hidden[-1]

        return self.rnn_hidden_to_latent(final_hidden) # N,q