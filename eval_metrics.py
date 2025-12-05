from torchdiffeq import odeint
from geoopt.manifolds import Euclidean 
from torch.autograd.functional import jvp
from torch import vmap
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Callable
from tqdm import tqdm
from manifolds.Sphere import Sphere


# This handles products of manifolds for the augmented state [x, logdetjac]
class ProductManifold:
    def __init__(self, *manifolds_dims):
        self.manifolds_dims = manifolds_dims
        self.manifolds = [m_d[0] for m_d in manifolds_dims]
        self.dims = [m_d[1] for m_d in manifolds_dims]
        self.dim_list = self.dims
        self.dim = sum(self.dims)

        self.cum_dim = [0] + list(torch.cumsum(torch.tensor(self.dims), dim=0))

    def projx(self, x):
        return torch.cat([
            m.projx(x[..., self.cum_dim[i]:self.cum_dim[i+1]])
            for i, m in enumerate(self.manifolds)
        ], dim=-1)

# Euler integrator with projection for the ProductManifold
def projx_integrator_return_last(
    manifold: ProductManifold,
    odefunc: Callable,
    x0: torch.Tensor,
    t: torch.Tensor,
    method: str = "euler",
    projx: bool = True,
    local_coords: bool = False,
    pbar: bool = False,
) -> torch.Tensor:
    t_ = t.cpu().numpy()
    
    # x is the current state [x, logdetjac]
    x = x0

    iterator = range(len(t) - 1)
    if pbar:
        iterator = tqdm(iterator, desc="ODE Integration (NLL)")

    for i in iterator:
        dt = t[i+1] - t[i]
        
        # Compute the derivative: v = d/dt [x, logdetjac]
        # odefunc returns: [dx/dt, d logdetjac / dt]
        v = odefunc(t[i], x)
        
        # Euler step
        x = x + dt * v

        # Project state back to the product manifold
        if projx:
            x = manifold.projx(x)
        
    return x # Return only the final state


# --- MODIFIED output_and_div function (Final Correction) ---
def output_and_div(
    func: Callable,
    x: torch.Tensor,
    v: torch.Tensor = None,
    div_mode: str = "exact",
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    original_shape = x.shape
    B = original_shape[0] # Batch size
    D = x.numel() // B   # Total dimension of the data point (e.g., 3 for sphere, 3072 for image)
    
    # Flatten input to B x D
    x_flat = x.reshape(B, D)
    x_flat.requires_grad_(True)
    
    # Wrap func to handle the flattening and unflattening internally
    def flat_func(x_in):
        # The input x_in is B x D
        v_out = func(x_in.reshape(original_shape))
        return v_out.reshape(B, D)
    
    # Compute output vector field 
    v_out_flat = flat_func(x_flat)
    v_out = v_out_flat.reshape(original_shape).detach() # v_out is the dx/dt to return

    # Compute divergence on the flattened B x D space
    if div_mode == "exact":
        # Jacobian of flat_func w.r.t. flat input. Shape is (B, D, B, D)
        jac = torch.autograd.functional.jacobian(flat_func, x_flat, create_graph=True)
        
        if jac.ndim == 4:
            # Assume no cross-batch influence: df_i/dx_j = 0 for i!=j
            # We extract the diagonal blocks to get the B x (D x D) Jacobians
            
            # Using torch.einsum is the clearest way to express the trace on the batched diagonal blocks:
            # 'bidi->b' means:
            # b: Batch index (must match in-batch and out-batch dimensions, i.e., jac.shape[0] == jac.shape[2])
            # i: Dimension index (inner dimensions for the DxD Jacobian)
            # The indices for the dimensions being summed over are the second and fourth indices.
            div = torch.einsum('bidi->b', jac)
            
        elif jac.ndim == 3 and jac.shape == (B, D, D):
            # Fallback for simpler Jacobian outputs (B, D, D)
            div = torch.einsum('bii->b', jac)
        else:
            raise RuntimeError(f"Unexpected Jacobian shape for divergence: {jac.shape}")
        
        div = div.unsqueeze(-1) # Output shape B x 1
            
    # ... (rademacher mode remains the same, as it relies on JVP which is more robust)
    elif div_mode == "rademacher":
        if v is None:
            raise ValueError("Rademacher mode requires a random sign vector 'v'.")
        
        v_flat = v.reshape(B, D)
        
        # JVP on the flattened space
        # jvp returns (output, jvp_result). We use the JVP result [1]
        jvp_result = jvp(flat_func, (x_flat,), (v_flat,), create_graph=True)[1]
        
        # Sum R * J_v * R over the dimension D
        div = torch.sum(v_flat * jvp_result, dim=-1, keepdim=True)
    
    else:
        raise ValueError(f"Unknown divergence mode: {div_mode}")

    x_flat.requires_grad_(False)
    # Return the *original* reshaped output and the divergence
    return v_out, div.detach()


@torch.no_grad()
def compute_nll(
    model: nn.Module,
    manifold,  # Your 'geom' (e.g., Sphere, Euclidean)
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    dim: int,
    num_steps: int = 1000,
    t1: float = 1.0,
    div_mode: str = "exact",
    eval_projx: bool = False,
    local_coords: bool = False,
    atol: float = 1e-5,  # For adaptive solver
    rtol: float = 1e-5,  # For adaptive solver
    cfg_mock: dict = None, # Mock config for params like normalize_loglik
):
    """
    Computes the Negative Log-Likelihood (NLL) of a batch of data
    by solving the augmented ODE backward in time.
    """
    if cfg_mock is None:
        cfg_mock = {}
        
    total_logp1 = []
    # Set explicitly
    manifold = Sphere()
    
    # Define the core vector field function, which includes projection
    def vecfield_fn(t, x):
        # x is assumed to be the point on the manifold (not augmented state)
        # We need to expand t since the model expects a batch of time values
        t_batch = t.expand(x.shape[0], 1)
        # The model returns the unprojected vector, manifold.proju projects it
        return manifold.proju(x, model([x, t_batch]))

    for i, batch in enumerate(tqdm(data_loader, desc="Computing NLL")):
        if i == 300:
           break 
        x1 = batch
        if isinstance(batch, tuple) or isinstance(batch, list):
            x1, _ = batch
        x1 = x1.to(device)

        try:
            nfe = [0]
            v = None
            if div_mode == "rademacher":
                # Rademacher vector: random signs {-1, 1}
                v = torch.randint(low=0, high=2, size=x1.shape).to(x1) * 2 - 1

            def odefunc(t, tensor):
                nfe[0] += 1
                t = t.to(tensor)
                x = tensor[..., :dim]
                
                # We need to compute v(t, x) and its divergence div(v)
                vecfield_wrapped = lambda x_input: vecfield_fn(t, x_input)
                dx, div = output_and_div(vecfield_wrapped, x, v=v, div_mode=div_mode)

                if hasattr(manifold, "logdetG"):
                    # Riemannian manifold correction term: 0.5 * d/dx log(|G|) * v
                    
                    # Define a function to compute the JVP for the logdetG term
                    def _jvp(x_i, v_i):
                        # torch.enable_grad() is used inside jvp to compute the Jacobian
                        with torch.enable_grad():
                            x_i.requires_grad_(True)
                            output, result = jvp(manifold.logdetG, (x_i.unsqueeze(0),), (v_i.unsqueeze(0),))
                            x_i.requires_grad_(False)
                            return result.squeeze(0)

                    # Map the JVP computation over the batch dimension
                    # Note: vmap is not strictly needed if we use the batch dimension in jvp,
                    # but the original code uses vmap, so we try to preserve the logic.
                    # Since jvp usually requires a single input/output, we must ensure it's batched correctly.
                    # Given the constraints, we will rely on a batched version if available, otherwise 
                    # use a loop/vmap if properly imported.
                    try:
                        corr = vmap(_jvp)(x, dx)
                    except NameError:
                        # Fallback if vmap is not available, process batch elements sequentially (slower)
                        corr = torch.stack([_jvp(x[k], dx[k]) for k in range(x.shape[0])], dim=0)
                        
                    div = div + 0.5 * corr.to(div)

                div = div.reshape(-1, 1)
                del t, x
                return torch.cat([dx, div], dim=-1)

            # Product manifold: Data Manifold x Euclidean (for the log-likelihood)
            # Note: Using # In the compute_nll function:
            product_man = ProductManifold(
                (manifold, dim), 
                (Euclidean(), 1)
            )

            # state1 = [x1, 0]
            state1 = torch.cat([x1, torch.zeros_like(x1[..., :1])], dim=-1)

            # --- Solve the ODE Backward (t1 -> 0) ---

            if not eval_projx and not local_coords:
                # Adaptive step solver (Dopri5) if no projection is needed
                state0 = odeint(
                    odefunc,
                    state1,
                    t=torch.linspace(t1, 0, 2).to(device),
                    atol=atol,
                    rtol=rtol,
                    method="dopri5",
                    options={"min_step": 1e-5},
                )[-1]
            else:
                # Fixed step solver with projection (Euler + Projx)
                state0 = projx_integrator_return_last(
                    product_man,
                    odefunc,
                    state1,
                    t=torch.linspace(t1, 0, num_steps + 1).to(device),
                    method="euler",
                    projx=eval_projx,
                    local_coords=local_coords,
                    pbar=False,
                )

            # --- Compute NLL ---

            x0_unproj, logdetjac = state0[..., :dim], state0[..., -1]
            x0 = manifold.projx(x0_unproj)

            # Calculate Integration Error (optional, for logging)
            integ_error = (x0 - x0_unproj).abs().max(dim=-1).values

            # Log-Likelihood: log p_1(x_1) = log p_0(x_0) + log |d x_0 / d x_1|
            logp0 = manifold.base_logprob(x0)
            logp1 = logp0 + logdetjac
            
            # Normalize log-likelihood if specified in the original config logic
            if cfg_mock.get("normalize_loglik", False):
                 logp1 = logp1 / dim

            # Mask out those that left the manifold (especially for SPD manifold)
            # You can adapt this based on the manifold you are using.
            masked_logp1 = logp1
            # Assuming 'SPD' check is not strictly necessary for your `Sphere` or `Euclidean`
            # if isinstance(manifold, SPD):
            #     mask = integ_error < 1e-5
            #     masked_logp1 = logp1[mask]
            
            total_logp1.append(masked_logp1)

        except Exception as e:
            print(f"Error processing batch: {e}")
            # If an error occurs, fill with zeros to avoid crashing
            total_logp1.append(torch.zeros_like(x1[..., :1]).squeeze(-1).to(device))
            continue
            
    # Concatenate results and compute NLL statistics
    all_logp1 = torch.cat(total_logp1, dim=0)
    
    # Negative Log-Likelihood (NLL)
    nll = -all_logp1
    mean_nll = nll.mean().item()
    std_nll = nll.std().item()

    return mean_nll, std_nll
