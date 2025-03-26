import torch
import torch.nn as nn


class HyperNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, r, init_zeros=False):
        super(HyperNetwork, self).__init__()
        self.n_layers = n_layers
        self.r = r
        self.stack = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        if init_zeros:
            self.layer_embeddings = torch.zeros(
                n_layers, input_dim
            ).requires_grad_(True).to(device)
        else:
            self.layer_embeddings = torch.randn(
                n_layers, input_dim
            ).requires_grad_(True).to(device)

    def forward(self, task_embedding):
        embedding_stack = torch.cat(
            [self.layer_embeddings, task_embedding.unsqueeze(0).expand(self.n_layers, -1)],
            dim=1,
        )
        output_params = self.stack(embedding_stack)
        output_params = output_params.view(self.n_layers, self.r, -1)
        return output_params
