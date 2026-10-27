"""
Adapted from https://github.com/lazaratan/meta-flow-matching/blob/main/src/models/components/mlp.py
"""

import torch


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x)


class torch_wrapper_flow_cond(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        padding = torch.zeros((x.shape[0], self.model.num_conditions)).long()
        out = self.model(t, x)
        return torch.cat([out, padding.to(x)], dim=1)
    

class torch_wrapper_gnn_flow_cond(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        padding = torch.zeros((x.shape[0], self.model.num_treat_conditions)).long()
        out = self.model(t, x)
        return torch.cat([out, padding.to(x)], dim=1)


class torch_wrapper_tv(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs): # 
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, *args, **kwargs):
        return self.net(x)

class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]


class MLP_cond(torch.nn.Module):
    """Class conditional, assume the last dimension(s) are cond"""
    def __init__(self, dim, cond=1, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.dim = dim
        self.conditional = int(cond)
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0) + int(cond), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, *args, **kwargs):
        out = self.net(x)
        # cat class conditional
        return torch.cat([out, x[:, -int(self.conditional):]], 1)
   
    
def mlp(input_size, output_size, hidden_size, num_layers, act_fn=torch.nn.SELU): #act_fn=torch.nn.SiLU
    layers = []
    prev_size = input_size
    for _ in range(num_layers-1):
        layers.append(torch.nn.Linear(prev_size, hidden_size))
        layers.append(act_fn())
        prev_size = hidden_size
    layers.append(torch.nn.Linear(hidden_size, output_size))
    return torch.nn.Sequential(*layers)


class SkipMLP(torch.nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size, num_layers, act_fn=torch.nn.SELU
    ):
        super(SkipMLP, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.act_fn = act_fn

        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        self.activation = act_fn()

    def forward(self, x):
        out = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.activation(layer(out) + out)  # Skip connection
        out = self.output_layer(out)
        return out
