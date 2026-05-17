import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    # 修改
    # 新的噪声调度
    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        
        n_timestep = schedule_opt['n_timestep']
        self.kappa = schedule_opt.get('kappa', 0.05)    
        p = schedule_opt.get('p', 0.5)
        eta_1 = schedule_opt.get('eta_1', 1e-5)
        eta_T = schedule_opt.get('eta_T', 1.0)
        
        assert n_timestep > 1
        assert eta_1 > 0 and eta_T > 0
        assert eta_T >= eta_1 # 我们希望噪声是从小到大递增的
        assert p > 0 # 

        b_0 = np.exp(1 / (2 * (n_timestep - 1)) * np.log(eta_T / eta_1))

        # 构造 /eta 序列
        self.num_timesteps = n_timestep
        eta = np.zeros(n_timestep + 1) # 初始化
        beta = np.zeros(n_timestep + 1)
        for t in range(1, n_timestep + 1):
            beta[t] = ((t - 1) / (n_timestep - 1)) ** p * (n_timestep - 1)
            eta[t] = eta_1 * (b_0 ** 2) ** beta[t]
        
        # \alpha_t
        alpha = np.zeros(n_timestep + 1)
        alpha[1:] = eta[1:] - eta[:-1]

        # 前向 buffer
        self.register_buffer('eta_t', to_torch(eta))
        self.register_buffer('alpha_t', to_torch(alpha))
        self.register_buffer('sigma_t', self.kappa * to_torch(np.sqrt(eta)))

        # 反向（后验）buffer
        posterior_var = np.zeros(n_timestep + 1)
        posterior_coef_x0 = np.zeros(n_timestep + 1)
        posterior_coef_xt = np.zeros(n_timestep + 1)

        for t in range(1, n_timestep + 1):
            posterior_coef_x0[t] = alpha[t] / eta[t]
            posterior_coef_xt[t] = eta[t-1] / eta[t]
            posterior_var[t] = (self.kappa**2) * eta[t-1] * alpha[t] / eta[t]

        self.register_buffer('posterior_var', to_torch(posterior_var))
        self.register_buffer('posterior_coef_x0', to_torch(posterior_coef_x0))
        self.register_buffer('posterior_coef_xt', to_torch(posterior_coef_xt))

    '''不需要了
    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise
    '''
    
    # 修改
    # 后验
    def q_posterior(self, x_start, x_t, y, t):
        
        posterior_mean = (
            extract(self.posterior_coef_xt, t, x_t.shape) * x_t +
            extract(self.posterior_coef_x0, t, x_t.shape) * x_start
        )

        posterior_var = extract(self.posterior_var, t, x_t.shape)
        return posterior_mean, posterior_var
        

    
    # 修改
    def p_mean_variance(self, x, t, y=None):
        batch_size = x.shape[0]
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x.device, dtype=torch.long)
        # 将eta_t作为时间信息传给网络
        eta_t_val = extract(self.eta_t, t, (batch_size, 1))

        if y is not None:
            x_recon = self.denoise_fn(torch.cat([y, x], dim=1), eta_t_val)
        else:
            x_recon = self.denoise_fn(x, eta_t_val)

        model_mean, model_var = self.q_posterior(
            x_start=x_recon, x_t=x, y=y, t=t
        )

        return model_mean, model_var

    #修改
    @torch.no_grad()
    def p_sample(self, x, t, y=None):
        model_mean, model_var = self.p_mean_variance(x=x, t=t, y=y)
        if t > 1:
            noise = torch.randn_like(x)
            return model_mean + noise * model_var.sqrt()
        else:
            return model_mean  # 最后一步不加噪声
    # 修改
    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.eta_t.device
        sample_inter = max(1, self.num_timesteps // 10)

        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(1, self.num_timesteps + 1)), 
                          desc='sampling loop',total=self.num_timesteps
                          ):
              img = self.p_sample(img, i)
              if i % sample_inter == 0:
                  ret_img = torch.cat([ret_img, img], dim=0)
        else:
            y = x_in  # LR 条件图
            shape = y.shape
            # 修改采样起点
            img = y + self.kappa * torch.randn(shape, device=device)
            ret_img = y
            for i in tqdm(reversed(range(1, self.num_timesteps + 1)),
                          desc='sampling loop', total=self.num_timesteps):
                img = self.p_sample(img, i, y=y)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    # 修改
    def q_sample(self, x_start, y, t, noise=None):
        e_0 = y - x_start
        noise = default(noise, lambda: torch.randn_like(x_start))

        curr_eta = extract(self.eta_t, t, x_start.shape)
        curr_sigma = extract(self.sigma_t, t, x_start.shape)

        return x_start + curr_eta * e_0 + curr_sigma * noise
    
    # 修改
    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        y = x_in['SR']
        b, c, h, w = x_start.shape

        # 均匀采样时间步
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=x_start.device)

        # 前向
        x_noisy = self.q_sample(x_start=x_start, y=y, t=t, noise=noise)

        eta_t_cond = extract(self.eta_t, t, (b, 1))

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, eta_t_cond)
        else:
            x_recon = self.denoise_fn(
                torch.cat([y, x_noisy], dim=1), eta_t_cond
            )
        
        loss = self.loss_func(x_recon, x_start)

        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
