# Taken from https://github.com/facebookresearch/riemannian-fm/blob/b6ac1e9d60e18e594fb6310ec8a68d9ae683e2b2/manifm/manifolds/sphere.py

import math
import torch
from geoopt.manifolds import Sphere as geoopt_Sphere


class Sphere(geoopt_Sphere):
    def transp(self, x, y, v):
        denom = 1 + self.inner(x, x, y, keepdim=True)
        res = v - self.inner(x, y, v, keepdim=True) / denom * (x + y)
        cond = denom.gt(1e-3)
        return torch.where(cond, res, -v)

    def uniform_logprob(self, x):
        dim = x.shape[-1]
        return torch.full_like(
            x[..., 0],
            math.lgamma(dim / 2) - (math.log(2) + (dim / 2) * math.log(math.pi)),
        )

    def random_base(self, *args, **kwargs):
        return self.random_uniform(*args, **kwargs)

    def base_logprob(self, *args, **kwargs):
        return self.uniform_logprob(*args, **kwargs)

