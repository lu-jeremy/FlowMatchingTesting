import torch
import torch.nn.functional as F
from torch import nn
import math
from typing import List


class SinusoidalTimeEmbedding(nn.Module):
    """
    Produce sinusoidal embeddings for scalar time inputs.
    Output shape: (B, dim)
    Accepts t shaped as: scalar, (B,), (B,1), or (B, 1) like tensors.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim >= 1, "dim must be >= 1"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Normalize t to shape (B, 1)
        if t.dim() == 0:
            t = t.reshape(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)  # (B,) -> (B,1)
        elif t.dim() >= 2 and t.size(-1) != 1 and t.size(-1) != self.dim:
            # if t is (B, something > 1) assume it's already okay or malformed;
            # we'll just try to collapse to (B,1) if possible
            t = t.reshape(t.shape[0], -1)[:, :1]

        device = t.device
        batch_size = t.shape[0]

        # Build frequency terms
        half_dim = self.dim // 2
        # scaling factor
        freq_log = math.log(10000.0) / max(half_dim - 1, 1)
        inv_freq = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -freq_log)

        # t: (B,1) -> (B, half_dim) after multiplication
        args = t.to(dtype=torch.float32) * inv_freq.unsqueeze(0)  # (B, half_dim)

        sin = torch.sin(args)
        cos = torch.cos(args)

        emb = torch.cat([sin, cos], dim=-1)  # (B, 2*half_dim)

        if self.dim % 2 == 1:
            # odd output dim -> pad one extra zero column
            emb = torch.cat([emb, torch.zeros(batch_size, 1, device=device, dtype=emb.dtype)], dim=-1)

        # Ensure exact dimensionality
        emb = emb[:, : self.dim]
        return emb


class MLP(nn.Module):
    """
    Simple MLP that conditions on a time embedding.
    Forward expects inputs: (x, t) where:
      - x: Tensor with shape (B, in_dim) or (B, C, H, W) where product equals in_dim
      - t: scalar, (B,), (B,1) or (B, out_time_dim) (if already embedded)
    """

    def __init__(
        self,
        in_dim: int,
        num_channels: int,
        h_dim: int = 64,
        enable_time_embed: bool = True,
        num_hidden: int = 2,
        out_time_dim: int = 2,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.num_channels = num_channels
        self.h_dim = h_dim
        self.num_hidden = num_hidden
        self.enable_time_embed = enable_time_embed
        self.out_time_dim = out_time_dim

        # First layer: concat input flattened vector with time embedding
        self.input_blocks = nn.Sequential(
            nn.Linear(self.in_dim + (self.out_time_dim if self.enable_time_embed else 0), self.h_dim),
            nn.SiLU(),
        )

        # Residual hidden blocks
        self.hidden_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.h_dim),
                    nn.Linear(self.h_dim, self.h_dim),
                    nn.SiLU(),
                )
                for _ in range(self.num_hidden)
            ]
        )

        # Final projection back to input dimension
        self.fc1 = nn.Sequential(
            nn.LayerNorm(self.h_dim),
            nn.Linear(self.h_dim, self.in_dim),
        )

        if self.enable_time_embed:
            self.time_embedding = SinusoidalTimeEmbedding(self.out_time_dim)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x, t = inputs  # type: (Tensor, Tensor)

        # Save original shape for reshaping back at the end
        orig_shape = x.shape  # e.g., (B, in_dim) or (B, C, H, W)
        batch_size = x.shape[0]

        # Flatten features to (B, in_dim)
        x_flat = x.view(batch_size, -1)
        assert x_flat.shape[1] == self.in_dim, f"Flattened input dim {x_flat.shape[1]} != in_dim {self.in_dim}"

        # Build time embedding of shape (B, out_time_dim)
        if self.enable_time_embed:
            # Accept t as scalar, (B,), (B,1) or already (B, out_time_dim)
            if t is None:
                raise ValueError("Time 't' must be provided when enable_time_embed=True")

            # If user passed an already-embedded vector with matching out_time_dim, use it
            if t.dim() == 2 and t.shape[-1] == self.out_time_dim:
                t_emb = t.to(dtype=x_flat.dtype)
                if t_emb.shape[0] != batch_size:
                    # expand or repeat if necessary
                    t_emb = t_emb.expand(batch_size, -1)
            else:
                # Normalize to (B,1) and run sinusoidal embedder
                if t.dim() == 0:
                    t_norm = t.reshape(1, 1).to(dtype=x_flat.dtype)
                    t_norm = t_norm.expand(batch_size, 1)
                elif t.dim() == 1:
                    # (B,) -> (B,1)
                    if t.shape[0] == batch_size:
                        t_norm = t.unsqueeze(-1).to(dtype=x_flat.dtype)
                    else:
                        # allow broadcasting from single value
                        t_norm = t.reshape(1, 1).expand(batch_size, 1).to(dtype=x_flat.dtype)
                else:
                    # (B, k) where k>1 but not out_time_dim -> take first column as time scalar
                    t_norm = t.reshape(batch_size, -1)[:, :1].to(dtype=x_flat.dtype)

                t_emb = self.time_embedding(t_norm)  # (B, out_time_dim)
        else:
            # no time embedding; create zero vector (or optionally accept t for other uses)
            t_emb = torch.zeros(batch_size, 0, device=x_flat.device, dtype=x_flat.dtype)

        # Concatenate input and time embedding
        x_in = torch.cat([x_flat, t_emb], dim=-1)  # (B, in_dim + out_time_dim)

        # Forward
        h = self.input_blocks(x_in)  # (B, h_dim)
        # Residual blocks
        for module in self.hidden_blocks:
            h = h + module(h)

        out_flat = self.fc1(h)  # (B, in_dim)

        # Reshape back to original input shape
        out = out_flat.view(*orig_shape)
        return out


# class SinusoidalTimeEmbedding(nn.Module):
#     def __init__(self, dim: int):
#         super().__init__()
#         self.dim = dim
#         
#     def forward(self, t: torch.Tensor) -> torch.Tensor:
#         device = t.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = t[:, None] * embeddings[None, :]
#         embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
#         return embeddings
# 
# 
# class MLP(nn.Module):
#     def __init__(
#             self,
#             in_dim: int,
#             num_channels: int,
#             h_dim: int = 64,
#             enable_time_embed: bool = True,
#             num_hidden: int = 2,
#             out_time_dim: int = 2,
#     ):
#         
#         super().__init__()
# 
#         # keep in mind that batch dimensions are implicit, and will heavily impact training time
#         self.in_dim = in_dim
#         self.out_time_dim = out_time_dim
#         self.h_dim = h_dim
#         self.num_hidden = num_hidden
#         # self.vgg_out_dim = 1000
#         self.num_channels = num_channels
#         self.enable_time_embed = enable_time_embed
# 
#         self.input_blocks = nn.Sequential(
#             nn.Linear(self.in_dim + self.out_time_dim, self.h_dim),
#             nn.SiLU(),
#         )
# 
#         self.hidden_blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.LayerNorm(self.h_dim),
#                 nn.Linear(self.h_dim, self.h_dim),
#                 nn.SiLU(),
#             )
#             for _ in range(self.num_hidden)
#         ])
# 
#         self.fc1 = nn.Sequential(
#             nn.LayerNorm(self.h_dim),
#             nn.Linear(self.h_dim, self.in_dim),
#             # nn.Tanh(),
#         )
# 
#         # self.linear_probe = nn.Sequential(
#         #     nn.Linear(self.vgg_out_dim, self.in_dim), 
#         #     nn.SiLU(),
#         # )
# 
#         # freeze vgg19 backbone
#         # self.vgg19 = vgg19(weights=VGG19_Weights.DEFAULT)
# 
#         # for param in self.vgg19.parameters():
#         #     param.requires_grad = False
# 
#         if self.enable_time_embed:
#             self.time_embedding = SinusoidalTimeEmbedding(self.out_time_dim)
#             # self.time_embedding = nn.Sequential(
#             #     nn.Linear(1, self.out_time_dim),
#             #     nn.SiLU(),
#             # )
# 
#     def forward(self, inputs: List[torch.Tensor, torch.Tensor]) -> torch.Tensor:
#         
#         x, t = inputs
#         size = x.size()
# 
#         # if self.num_channels == 3:
#         #     x = self.vgg19(x)
#         #     x = self.linear_probe(x)
#         
#         x = x.view(-1, self.in_dim)
# 
#         if self.enable_time_embed:
#             t = self.time_embedding(t)
#             
#             # in the case of 1 item batch
#             if len(t.size()) <= 1:
#                 t = t.unsqueeze(0)
# 
#         # ODE solver only allows 1D time trajectory inputs
#         t = t.expand(len(x), -1)
#     
#         if t.dim() <= 1:
#             t = t.unsqueeze(-1)
#         
#         # concatenate works better than add
#         x = self.input_blocks(torch.cat((x, t), dim=-1))
# 
#         for module in self.hidden_blocks:
#             x = x + module(x)
#             # x = F.silu(x)
# 
#         x = self.fc1(x)
#         return x.reshape(*size)
# 
# 
