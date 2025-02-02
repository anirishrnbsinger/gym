import torch
from torch.optim import Optimizer
import math

class COCOB_Backprop(Optimizer):
    """
    COntinuous COin Betting Backprop optimizer (Orabona & Tommasi, 2017)
    Implements parameter-free optimization using coin betting framework
    
    Key Components:
    - Maintains per-parameter state variables
    - Automatically adapts effective learning rates
    - No manual learning rate tuning required
    
    Args:
        params (iterable): Iterable of parameters to optimize
        alpha (float): Scaling factor (default: 100)
        verbose (bool): Print optimization statistics (default: False)
    """
    
    def __init__(self, params, alpha=100, verbose=False):
        defaults = dict(alpha=alpha, verbose=verbose)
        super().__init__(params, defaults)
        
        # Initialize tracking variables
        self.step_count = 0
        print(f"COCOB initialized with alpha={alpha}, Verbose={'On' if verbose else 'Off'}")

    def step(self, closure=None):
        """
        Performs a single optimization step
        
        State Variables for Each Parameter Tensor:
        - G: Sum of absolute gradient values (L1 norm)
        - reward: Cumulative reward from betting strategy
        - theta: Sum of gradients (similar to momentum)
        - prev_w: Initial parameter values
        """
        
        loss = None
        if closure is not None:
            loss = closure()

        # Print header every 10 steps
        verbose = self.param_groups[0]['verbose']
        if verbose and self.step_count % 10 == 0:
            print(f"\n{'Step':<6} {'Param':<15} {'Grad Norm':<12} {'Beta':<12} {'Update':<12}")
            print("-" * 60)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]

                # Initialize state on first step [2][10]
                if len(state) == 0:
                    state['step'] = 0
                    state['G'] = torch.zeros_like(p.data) + 1e-8  # L_i in paper [15]
                    state['reward'] = torch.zeros_like(p.data)
                    state['theta'] = torch.zeros_like(p.data)
                    state['prev_w'] = p.data.clone()
                    if verbose:
                        print(f"Initialized parameter tensor of shape {p.data.shape}")

                G = state['G']
                reward = state['reward']
                theta = state['theta']
                prev_w = state['prev_w']
                alpha = group['alpha']
                
                # Algorithm 1 from paper (lines 6-10) [15]
                # Update tracked parameters
                state['G'] = G + grad.abs()
                state['reward'] = reward + (p.data - prev_w) * grad
                state['theta'] = theta + grad
                
                # Compute betting fraction (β)
                scaled_theta = 2 * theta / (G + 1e-8)
                sigma = 1 / (1 + torch.exp(-scaled_theta))  # Sigmoid [15]
                beta = (2 * sigma - 1) / (G * alpha + 1e-8)
                
                # Parameter update rule
                new_w = state['prev_w'] + beta * (alpha * G + reward)
                p.data.copy_(new_w)
                
                # Print statistics
                if verbose:
                    grad_norm = grad.norm().item()
                    beta_mean = beta.mean().item()
                    update_size = (new_w - prev_w).norm().item()
                    
                    print(f"{self.step_count:<6} {str(p.shape):<15} "
                          f"{grad_norm:<12.4f} {beta_mean:<12.4e} {update_size:<12.4f}")

                state['prev_w'] = new_w.clone()
                state['step'] += 1

        self.step_count += 1
        return loss
