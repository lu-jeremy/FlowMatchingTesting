import torch
from typing import Tuple


class Path:
    def __init__(self, sigma, path_type):
        self.sigma = sigma
        self.path_type = path_type

    def sample_path(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        geom = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns sample and target vector field.

        return:
            x_t (torch.Tensor): Diffused sample
            target (torch.Tensor): Target vector field
        '''
        # dependent on user input
        if len(t.size()) != len(x_1.size()):
            t = t[:, None, None].expand(x_1.shape)

        # \mu_{t} = tx_{1}, \sigma_{t} = 1 - (1 - \sigma_{\min})t
        if self.path_type == 'CFM':
            x_t = (1. - (1. - self.sigma) * t) * torch.randn_like(x_1) + t * x_1
            target = (x_1 - (1. - self.sigma) * x_t) / (1. - (1. - self.sigma) * t)
        elif self.path_type == 'iCFM':
            x_t = (1. - t) * x_0 + t * x_1 + self.sigma * torch.randn_like(x_1)
            target = x_1 - x_0
        elif self.path_type == 'Geodesic':
            # Vector on x_1's tangent plane
            dgradx = geom.logmap(x_0, x_1)
            x_t = geom.expmap(x_0, t * dgradx)
            target = geom.proju(x_t, dgradx)

        return x_t, target
