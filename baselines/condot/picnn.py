"""
Adapted from https://github.com/bunnech/condot/blob/main/condot/networks/npicnn.py
"""

import numpy as np
import torch
from torch import autograd
from torch import nn

from baselines.condot.layers import NonNegativeLinear, PosDefPotentials

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}


class PICNN(nn.Module):
    def __init__(
        self,
        input_dim,
        input_dim_label,
        hidden_units,
        activation="leakyrelu",
        softplus_wz_kernels=False,
        softplus_beta=1,
        fnorm_penalty=0,
        kernel_init_fxn=None,
        neural_embedding=None,
        **kwargs
    ):

        super(PICNN, self).__init__()
        self.fnorm_penalty = fnorm_penalty
        self.softplus_wz_kernels = softplus_wz_kernels

        if isinstance(activation, str):
            activation = ACTIVATIONS[activation.lower().replace("_", "")]
        self.sigma = activation

        self.input_dim = input_dim
        self.input_dim_label = input_dim_label

        # neural embedding encoding
        self.neural_embedding = neural_embedding
        if self.neural_embedding is not None:
            self.we = nn.Linear(in_features=self.neural_embedding[0],
                                out_features=self.neural_embedding[1])

        if self.neural_embedding is not None:
            effective_label_dim = self.neural_embedding[1]
        else:
            effective_label_dim = input_dim_label

        # compute identity maps for initialization
        factors, means = self.compute_identity_maps(input_dim, effective_label_dim)

        units = [effective_label_dim] + hidden_units

        self.n_layers = len(units)

        if self.softplus_wz_kernels:
            def Linear(*args, **kwargs):
                return NonNegativeLinear(*args, **kwargs, beta=softplus_beta)
            # this function should be inverse map of function used in PositiveDense layers
            rescale = lambda x: np.log(np.exp(x) - 1)
        else:
            Linear = nn.Linear
            rescale = lambda x: x

        # self layers for hidden state u, when contributing all ~0
        self.w = list()
        for idim, odim in zip([effective_label_dim] + [units[0]] + units[:-2],
                              [units[0]] + units[:-1]):
            _w = nn.Linear(idim, odim, bias=True)
            if kernel_init_fxn is not None:
                kernel_init_fxn(_w.weight)
            nn.init.zeros_(_w.bias)
            self.w.append(_w)
        self.w = nn.ModuleList(self.w)

        # first layer for hidden state performs a comparison with database
        # for arbitrary labels, we use default initialization (not identity)
        self.w_0 = nn.Linear(effective_label_dim, effective_label_dim, bias=False)

        # auto layers for z, should be mean operators with no bias
        # keep track of previous size to normalize accordingly
        normalization = 1
        self.wz = list()
        for idim, odim in zip(units, units[1:] + [1]):
            _wz = Linear(idim, odim, bias=False)
            nn.init.constant_(_wz.weight, rescale(1.0 / normalization))

            self.wz.append(_wz)
            normalization = odim
        self.wz = nn.ModuleList(self.wz)

        # for family of convex functions stored in z, if using init then first
        # vector z_0 has as many values as # of convex potentials.
        self.w_z0 = PosDefPotentials(self.input_dim, effective_label_dim, bias=True)
        with torch.no_grad():
            self.w_z0.weight.copy_(factors)
            self.w_z0.bias.copy_(means)

        # cross layers for convex functions z / hidden state u
        # initialized to be identity first with 0 bias
        # and then ~0 + 1 bias to ensure identity
        self.wzu = list()

        _wzu = nn.Linear(effective_label_dim, units[0], bias=True)
        if kernel_init_fxn is not None:
            kernel_init_fxn(_wzu.weight)
        nn.init.constant_(_wzu.bias, rescale(1.0))
        self.wzu.append(_wzu)
        zu_layers = zip(units[1:], units[1:])

        for idim, odim in zu_layers:
            _wzu = nn.Linear(idim, odim, bias=True)
            if kernel_init_fxn is not None:
                kernel_init_fxn(_wzu.weight)
            nn.init.constant_(_wzu.bias, rescale(1.0))
            self.wzu.append(_wzu)
        self.wzu = nn.ModuleList(self.wzu)

        # self layers for x, ~0
        self.wx = list()
        for odim in (units + [1]):
            _wx = nn.Linear(input_dim, odim, bias=True)
            if kernel_init_fxn is not None:
                kernel_init_fxn(_wx.weight)
            nn.init.zeros_(_wx.bias)
            self.wx.append(_wx)
        self.wx = nn.ModuleList(self.wx)

        # cross layers for x / hidden state u, all ~0
        self.wxu = list()
        for idim in ([units[0]] + units):
            _wxu = nn.Linear(idim, input_dim, bias=True)
            if kernel_init_fxn is not None:
                kernel_init_fxn(_wxu.weight)
            nn.init.zeros_(_wxu.bias)
            self.wxu.append(_wxu)
        self.wxu = nn.ModuleList(self.wxu)

        # self layers for hidden state u, to update z, all ~0
        self.wu = list()
        for idim, odim in zip([units[0]] + units, units + [1]):
            _wu = nn.Linear(idim, odim, bias=False)
            if kernel_init_fxn is not None:
                kernel_init_fxn(_wu.weight)
            self.wu.append(_wu)
        self.wu = nn.ModuleList(self.wu)

    def forward(self, x, y):
        if y.ndim == 1:
            y = y.repeat(x.size(dim=0), 1)

        if self.neural_embedding:
            y = nn.Sigmoid()(self.we(y))

        # initialize u and z
        u = y

        u = nn.functional.softmax(10 * self.w_0(u), dim=1)
        z_0 = self.w_z0(x)
        z = self.sigma(0.2)(z_0 * u)
        # apply k layers - 1
        for i in range(1, self.n_layers):
            u = self.sigma(0.2)(self.w[i](u))
            t_u = nn.functional.softplus(self.wzu[i - 1](u))
            z = self.sigma(0.2)(
                self.wz[i - 1](torch.mul(z, t_u))
                + self.wx[i](torch.mul(x, self.wxu[i](u))) + self.wu[i](u)
            )

        z = (self.wz[-1](torch.mul(z, nn.functional.softplus(self.wzu[-1](u))))
             + self.wx[-1](torch.mul(x, self.wxu[-1](u))) + self.wu[-1](u))
        return z

    def transport(self, x, y):
        assert x.requires_grad

        (output,) = autograd.grad(
            self.forward(x, y),
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size()[0], 1), device=x.device).float(),
        )
        return output

    def clamp_w(self):
        if self.softplus_wz_kernels:
            return

        for w in self.wz:
            w.weight.data = w.weight.data.clamp(min=0)
        return

    def penalize_w(self):
        return self.fnorm_penalty * sum(
            map(lambda x: torch.nn.functional.relu(-x.weight).norm(), self.wz)
        )

    def compute_identity_maps(self, input_dim, input_dim_label):
        A = torch.eye(input_dim).reshape((1, input_dim, input_dim))
        factors = A.repeat(input_dim_label, 1, 1)
        means = torch.zeros((input_dim_label, input_dim))
        return factors, means
