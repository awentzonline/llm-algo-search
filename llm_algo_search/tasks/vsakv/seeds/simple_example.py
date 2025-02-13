"""Simple VSA example using hadamard product for binding"""
import math

import torch


class VSA:
    def initialize(self, num_vecs, vec_dims):
        return torch.randn(num_vecs, vec_dims) / math.sqrt(vec_dims)

    def bundle(self, a, b):
        return a + b

    def bind(self, a, b):
        return a * b

    def unbind(self, r, x):
        return r / x

    def similarity(self, r, x):
        # r.shape is (batch, vec_dims)
        # x.shape is (batch, num_examples, vec_dims) for checking many vectors at the same time
        # output shape should be (batch, num_examples)
        # the element with the max value is chosen as present but non-present vectors
        # should be value <= 0
        r = r / (torch.norm(r, dim=-1, keepdim=True) + 1e-8)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(r, x.mT)