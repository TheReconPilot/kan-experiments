import torch
import torch.nn as nn

def larctan(x, k):
    return k * torch.atan(x)

class SKANLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, basis_function=None, device='cpu'):
        super(SKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.basis_function = basis_function
        self.device = device
        if bias:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features + 1).to(device))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        x = x.view(-1, 1, self.in_features)
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)
        y = self.basis_function(x, self.weight)
        y = torch.sum(y, dim=2)
        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class MultSKANNetwork(nn.Module):
    def __init__(self, n_a_list, n_m_list, basis_function=None, bias=True, device='cpu'):
        """
        n_a_list: list of addition widths [n_a0, n_a1, n_a2, ...]
        n_m_list: list of multiplication widths [n_m0, n_m1, n_m2, ...]
        """
        super(MultSKANNetwork, self).__init__()
        assert len(n_a_list) == len(n_m_list), "n_a_list and n_m_list must have the same length"
        self.n_a_list = n_a_list
        self.n_m_list = n_m_list
        self.device = device

        self.layers = nn.ModuleList()
        for l in range(len(n_a_list) - 1):
            in_features = n_a_list[l] + n_m_list[l]
            out_features = n_a_list[l+1] + 2 * n_m_list[l+1]
            self.layers.append(SKANLinear(in_features, out_features, bias=bias, 
                                          basis_function=basis_function, device=device))

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            x = layer(x)  # pass through SKANLinear
            n_a_next = self.n_a_list[l+1]
            n_m_next = self.n_m_list[l+1]

            if n_m_next > 0:
                # Split x into additive and multiplicative parts
                x_add = x[:, :n_a_next]  # First n_a_next elements stay the same
                x_mul_raw = x[:, n_a_next:]  # Remaining elements to be multiplied
                # Pairwise multiplication
                x_mul = x_mul_raw[:, 0::2] * x_mul_raw[:, 1::2]
                # Concatenate additive + multiplicative outputs
                x = torch.cat([x_add, x_mul], dim=1)
        return x
