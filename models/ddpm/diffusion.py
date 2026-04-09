import math
import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)

class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule='linear'):
        self.timesteps = timesteps
        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, timesteps, dtype=torch.float64)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(
                f"Unsupported beta schedule '{beta_schedule}'. "
                "Use 'linear' or 'cosine'."
            )
        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        out = a.to(t.device).gather(0, t).float()
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        return (self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        pred_noise = model(x_t, t)
        x_recon = (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                   self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise)
        x_recon = torch.clamp(x_recon, min=-1., max=1.)
        
        model_mean = (self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_recon +
                      self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
