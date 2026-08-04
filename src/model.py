import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim=86,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Compress encoder states
        self.hidden_fc = nn.Linear(hidden_dim, latent_dim)
        self.cell_fc = nn.Linear(hidden_dim, latent_dim)

        # Recover decoder states
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            input_dim
        )

        # Learnable SOS token
        self.start_token = nn.Parameter(
            torch.zeros(1, 1, input_dim)
        )

    def encode(self, x):
        _, (hidden, cell) = self.encoder(x)
        hidden_last = hidden[-1]
        cell_last = cell[-1]
        z_hidden = self.hidden_fc(hidden_last)
        z_cell = self.cell_fc(cell_last)
        return z_hidden, z_cell

    def decode(
        self,
        z_hidden,
        z_cell,
        seq_len,
    ):
        batch_size = z_hidden.size(0)
        hidden = self.latent_to_hidden(z_hidden)
        cell = self.latent_to_cell(z_cell)
        hidden = (
            hidden
            .unsqueeze(0)
            .repeat(self.num_layers, 1, 1)
        )
        cell = (
            cell
            .unsqueeze(0)
            .repeat(self.num_layers, 1, 1)
        )
        decoder_input = self.start_token.repeat(
            batch_size,
            1,
            1,
        )
        outputs = []
        hidden_state = (hidden, cell)
        for _ in range(seq_len):
            out, hidden_state = self.decoder(
                decoder_input,
                hidden_state,
            )
            prediction = self.output_layer(out)
            outputs.append(prediction)
            decoder_input = prediction
        reconstruction = torch.cat(
            outputs,
            dim=1,
        )
        return reconstruction

    def forward(self, x):
        seq_len = x.size(1)
        z_hidden, z_cell = self.encode(x)
        reconstruction = self.decode(
            z_hidden,
            z_cell,
            seq_len,
        )
        return reconstruction