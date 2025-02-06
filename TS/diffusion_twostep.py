# Standard Lib
from functools import partial
import os
import shutil
from glob import glob
import random

# Third Party
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from lion_pytorch import Lion
from torch.optim import AdamW, lr_scheduler
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from einops import rearrange, reduce, repeat
from PIL import Image
import pytorch_lightning as pl

# Local Imports
from utils.module_util import default
from utils.color_space import img2res, res2img, hslimg2res, hslres2img, img_normalize, img_unnormalize
from utils.module_util import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from utils.cond_utils import histro_equalize, cond_data_transforms
from utils.utils import pad_to_multiple, unpad_from_multiple

# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def on_cond_or_center_selector(img_lr, img_c, config_on_center_or_cond, histro=None):
    if config_on_center_or_cond == 'img_c':
        output = img_c
    elif config_on_center_or_cond == 'img_lr':
        output = img_lr
    else:
        NotImplementedError()
    
    if histro is not None:
        zeta = histro_equalize(img_lr, histro)
        output = (1-zeta)*output + zeta
        
    return output

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

def get_beta_schedule(num_diffusion_timesteps, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

class EnlightDiffusion(nn.Module):
    def __init__(self, unet, config) -> None:
        super().__init__()
        timesteps = config.timesteps
        self.sample_tqdm = True
        self.config = config
        self.unet = unet

        if config.on_res:
            if config.color_space == 'rgb':
                self.img2res = img2res
                self.res2img = res2img
            elif config.color_space == 'hsl':
                self.img2res = hslimg2res
                self.res2img = hslres2img
            else:
                NotImplementedError()
        else:
            if config.color_space == 'rgb':
                self.img2res = img_normalize
                self.res2img = img_unnormalize

            else:
                NotImplementedError()
        # create schedule
        betas = cosine_beta_schedule(timesteps, s=0.008)

        # create alpha_t_bar
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        extra_term_coef1 = (1-np.sqrt(alphas_cumprod))/(np.sqrt(1- alphas_cumprod))
        extra_term_coef2 = ((1-np.sqrt(alphas_cumprod_prev))*
                      (1-alphas) - (np.sqrt(alphas))*(1-np.sqrt(alphas))*(1-alphas_cumprod_prev))/(1-alphas_cumprod)
        
        self.num_timesteps = int(timesteps)

        # partial function to move to cuda
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # to insert in model.pth
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('extra_term_coef1', to_torch(extra_term_coef1))
        self.register_buffer('extra_term_coef2', to_torch(extra_term_coef2))

    def q_sample(self, x_start, t, center, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start)) # if no noise specified, we create noise
        if self.config.use_center:
            noise += extract(self.extra_term_coef1, t, x_start.shape)*normalize_to_neg_one_to_one(center)  # XXX: here we change to [-1,1]
        # t must be >=0
        t_cond = (t[:, None, None, None] >= 0).float()
        # if t <0, we force to be 0
        t = t.clamp_min(0)

        sqrt_alpha_t_bar = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_t_bar = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # return x_t if t>0, else return x_start
        return (
            sqrt_alpha_t_bar*x_start + sqrt_one_minus_alpha_t_bar*noise
        ) * t_cond + x_start*(1-t_cond)
    
    def p_losses(self, x_start, t, cond, center, feats_cond, noise=None):
        """
        [-1, 1] : x_start, noise
        [0, 1] : cond, center
        """
        x_t = self.q_sample(x_start, t, center, noise=noise)

        noise_pred = self.unet(x_t, t, cond, feats_cond=feats_cond)

        return noise_pred, noise 
    
    def forward(self, x_start, t, cond, center, feats_cond, noise=None):
        # create random noise
        noise = default(noise, lambda: torch.randn_like(x_start))
        noise_pred, noise = self.p_losses(x_start, t, cond, center, feats_cond, noise=noise)

        return noise_pred, noise

    @torch.no_grad()
    def sample(self, img_lr, encoder, histroEncoder=None, return_all_timesteps=False):
        if self.config.sample_mode == 'ddpm':
            sample_fn = self.p_sample_loop
        else:
            NotImplementedError(f"sample mode {self.config.sample_mode} not implemented")
        return sample_fn(img_lr, encoder, histroEncoder, return_all_timesteps)
    
    def p_sample_loop(self, img_lr, encoder, histroEncoder, return_all_timesteps = False):
        if return_all_timesteps:
            assert img_lr.shape[0] == 1, "return_all_timesteps only works with batch size 1"
            
        with torch.no_grad():
            img_c, feats_cond = encoder(img_lr)

        cond = on_cond_or_center_selector(img_lr, img_c, self.config.on_cond)

        center = on_cond_or_center_selector(img_lr, img_c, self.config.on_center, histroEncoder)
            
        if self.config.use_center_sampler:
            x_t = normalize_to_neg_one_to_one(center) #XXX: here we change to [-1,1], x_t not adding noise 
        else:
            # if no noise specified, we create noise
            x_t = torch.randn_like(img_lr) # the x_T
        
        def _get_timestep_iterator():
            it = reversed(range(0, self.num_timesteps))
            if self.sample_tqdm:
                it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)
            return it

        def _p_sample_loop_with_intermediates(x_t):            
            it = _get_timestep_iterator()

            for t in it:
                x_t, x_recon = self.p_sample(x_t, t, cond, img_lr, center, feats_cond=feats_cond, clip_denoised=self.config.clip_denoised)
                img_ = self.res2img(x_recon, img_lr, rescale_ratio=self.config.rescale_ratio)
                torch.cuda.empty_cache()
                yield img_

        def _p_sample_loop_final(x_t):            
            it = _get_timestep_iterator()

            for t in it:
                x_t, x_recon = self.p_sample(x_t, t, cond, img_lr, center, feats_cond=feats_cond, clip_denoised=self.config.clip_denoised)

            img = self.res2img(x_recon, img_lr, rescale_ratio=self.config.rescale_ratio)
            return img
        
        if return_all_timesteps:
            assert img_lr.shape[0] == 1, "return_all_timesteps only works with batch size 1"
            return _p_sample_loop_with_intermediates(x_t)
        else:
            return _p_sample_loop_final(x_t)

    def p_sample(self, x_t, t, cond, img_lr, center, feats_cond, noise_pred=None, clip_denoised = True, repeat_noise=False):
        """
        the reverse process to find q(x_t-1 |x_t) 
        """
        b, *_, device = *x_t.shape, x_t.device
        batched_times = torch.full((b,), t, device = x_t.device, dtype = torch.long)

        if noise_pred is None:
            noise_pred = self.unet(x_t, batched_times, cond=cond, feats_cond=feats_cond)

        model_mean, _, model_log_variance, x0_pred = self.p_mean_variance(
            x_t, batched_times, noise_pred, center, clip_denoised=clip_denoised)
        
        noise = noise_like(x_t.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (batched_times == 0).float()).reshape(b, *
                ((1,) * (len(x_t.shape) - 1)))  # if t is zero, dont add variance
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp()  * noise, x0_pred #sqrt(variance)
    
    def p_mean_variance(self, x_t, t, noise_pred, center, clip_denoised = True):
        x_recon = self.predict_start_from_noise(x_t, t, noise_pred)
        
        if clip_denoised:
            x_recon.clamp_(-1,1) #underline for inplace operation

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_recon, x_t, t, center)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def q_posterior(self, x_start, x_t, t, center):
        """
        to find the means for p(x_t-1, x_t)
        """
        
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        ) 

        if self.config.use_center_sampler:
            posterior_mean += extract(self.extra_term_coef2, t, x_t.shape) * normalize_to_neg_one_to_one(center) #XXX: here we change to [-1,1]

        # at T=0, posterior mean = x_start
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)

        # XXX: Why need this?
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        # to return u(x_t) , as mentioned in my blog
        return ( 
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
class LitDiffusion(pl.LightningModule):
    def __init__(self, diffusion_model, encoder, config): 
        super().__init__()
        self.save_hyperparameters(config) # now we can call self.hparams 
        self.model = diffusion_model

        self.encoder = encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.hparams.diffusion_path != '':
            from cond import LitHE, Unet_he
            unet_he = Unet_he(config, config.in_dim, True)

            self.histroEncoder = LitHE.load_from_checkpoint(
                config.diffusion_path, he_model=unet_he, encoder=encoder, config=config, strict=False).eval()
            for param in self.histroEncoder.parameters():
                param.requires_grad = False
                    
        self.auto_normalize = True

    def _on_valid_test_start(self):
        self.loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(self.device).eval() #input in domain [-1,1]
        for param in self.loss_fn_alex.parameters():
            param.requires_grad = False

    def on_validation_start(self) -> None:
        self._on_valid_test_start()

    def on_test_start(self) -> None:
        self._on_valid_test_start()
    
    def on_train_start(self) -> None:
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=self.hparams.lpips_type, normalize=True).to(
            self.device)  # normalize = False to expect input in domain [-1,1]

    def on_train_end(self):
        del self.lpips 

    def configure_optimizers(self):
        if self.hparams.optimizer == 'Lion':
            optimizer = Lion(self.model.parameters(), lr = self.hparams.train_lr)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.model.parameters(), lr = self.hparams.train_lr) # only affect on self.model
        else:
            NotImplementedError()
        
        if self.hparams.scheduler == 'plateau':
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',min_lr = self.hparams.min_lr, factor=self.hparams.factor, patience=self.hparams.patience),
                "monitor": "valid/combined",
                "frequency": 1,
                "interval": "epoch"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
                },
            }
        elif self.hparams.scheduler == 'cosine':
            from utils.scheduler import CosineWarmupScheduler
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters),
                },
            }
        elif self.hparams.scheduler is None: 
            return optimizer
        else:
            NotImplementedError("scheduler is not well defined")
        
    def loss_fn(self, noise_pred, noise, x_start, t, img_lr, center):
        def loss_dfpl(x_start, t, center, noise_pred, noise):
            # return the x_t_pred given predicted noise
            x_t_pred = self.model.q_sample(x_start, t, center, noise=noise_pred)
            x_t = self.model.q_sample(x_start, t, center, noise=noise)

            x_t_image = self.model.res2img(
                x_t, img_lr, rescale_ratio=self.hparams.rescale_ratio)
            x_t_image_pred = self.model.res2img(
                x_t_pred, img_lr, rescale_ratio=self.hparams.rescale_ratio)

            self.lpips.to(img_lr.device)
            # should be grad_enabled
            loss = self.lpips(x_t_image_pred, x_t_image)
            return loss
        
        if self.hparams.loss_type == 'l1':
            loss = F.smooth_l1_loss(noise_pred, noise)
        elif self.hparams.loss_type == 'l2':
            # loss = F.mse_loss(noise_pred, noise)
            loss = F.mse_loss(noise_pred, noise)
        elif self.hparams.loss_type == 'lpips':
            loss = loss_dfpl(x_start, t, center, noise_pred, noise)
        else:
            raise NotImplementedError()
        return loss

    def training_step(self, batch, batch_idxs):
        img_lr, img_hr, img_lr_name = batch
        t = None
        b, *_, device = *img_lr.shape, img_lr.device
        t = torch.randint(0, self.model.num_timesteps, (b,), device=device).long() \
            if t is None else torch.LongTensor([t]).repeat(b).to(device)

        x_start = self.model.img2res(
            img_hr, img_lr)  # to make


        with torch.no_grad():
            img_c, feats_cond = self.encoder(img_lr)

        cond = on_cond_or_center_selector(img_lr, img_c, self.hparams.on_cond)

        center = on_cond_or_center_selector(img_lr, img_c, self.hparams.on_center, self.histroEncoder)

        noise_pred, noise = self.model(x_start, t, cond, center, feats_cond=feats_cond)

        p_losses = self.loss_fn(noise_pred, noise, x_start, t, img_lr, center)

        self.log('train/p_losses', p_losses, prog_bar=True, on_step= True, logger=True, sync_dist=True)

        return p_losses
    
    def sample(self, img_lr, return_all_timesteps=False):
        # img_lr is already in [-1,1]
        imgs = self.model.sample(img_lr, self.encoder, self.histroEncoder , return_all_timesteps)  
        return imgs
    
    def _on_valid_test_epoch_start(self):
        if self.trainer.strategy.is_global_zero:
            if os.path.exists(self.hparams.results_folder):
                shutil.rmtree(self.hparams.results_folder)  # Delete folder and all contents
                
                # Make results folder 
            os.makedirs(self.hparams.results_folder)
        self.trainer.strategy.barrier()

    def on_validation_epoch_start(self) -> None:
        self._on_valid_test_epoch_start()

    def on_test_epoch_start(self):
        self._on_valid_test_epoch_start()
    
    def validation_epoch_end(self, validation_step_outputs) -> None:
        all_outputs = self.all_gather(validation_step_outputs)
        # 
        all_psnr = []
        all_ssim = []
        all_lpips = []
        all_ssim_gray = []

        for outputs in all_outputs:
            all_psnr.extend(outputs.get('list_psnr', []))
            all_ssim.extend(outputs.get('list_ssim', []))
            all_lpips.extend(outputs.get('list_lpips', []))
            all_ssim_gray.extend(outputs.get('list_ssim_gray', []))

        psnr_score = torch.mean(torch.stack(all_psnr)).item()
        ssim_score = torch.mean(torch.stack(all_ssim)).item()
        lpips_score = torch.mean(torch.stack(all_lpips)).item()
        ssim_gray_score = torch.mean(torch.stack(all_ssim_gray)).item()

        combined_score = (1 - lpips_score) * 0.8 + ssim_score * 0.1 + psnr_score * 0.1

        res = {
            'valid/psnr': np.around(psnr_score, 3),
            'valid/ssim': np.around(ssim_score, 3),
            'valid/ssim_gray': np.around(ssim_gray_score, 3),
            'valid/lpips': np.around(lpips_score, 3),
            'valid/combined': combined_score,
        }
        self.log_dict(res, prog_bar=True, logger=True, sync_dist=True) # we need to sync dist to make sure all gpus have the same results, which is for taking best model
        self.trainer.strategy.barrier()
        
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        self.trainer.strategy.barrier()

    def validation_step(self, batch, batch_idxs):
        # Each card will take a mini-batch
        img_lr, img_hr, img_lr_name = batch # img_hr domain [0,1], img_lr [0,1]

        original_shape = img_lr.shape

        if self.hparams.padding:
            img_lr = pad_to_multiple(img_lr)

        img_lr_name = list(img_lr_name)
        imgs = self.sample(img_lr, self.hparams.return_all_timesteps) 
        
        if self.hparams.padding:
            imgs = unpad_from_multiple(imgs, original_shape)

        list_psnr = []
        list_ssim = []
        list_lpips = []
        list_ssim_gray = []

        for i in range(imgs.shape[0]):
            # Convert tensor to numpy array and change from B C H W to B H W C
            sr = rearrange(imgs[i].detach().cpu().clamp(0, 1).numpy() * 255, 'c h w -> h w c')
            hr = rearrange(img_hr[i].detach().cpu().clamp(0, 1).numpy() * 255, 'c h w -> h w c')
            
            if hr.shape != sr.shape:
                from compare.full_ref import center_crop
                hr = center_crop(hr, sr)
            
            # metric calculation
            from compare.full_ref import _psnr, _ssim, _lpips, _ssim_gray, transform
            psnr_num = _psnr(hr, sr)
            ssim_num = _ssim(hr, sr)
            ssim_gray_num = _ssim_gray(hr, sr)

            sr_t = imgs[i].unsqueeze(0)  # Add batch dimension
            hr_t = img_hr[i].unsqueeze(0)  # Add batch dimension
            lpips_num = _lpips(hr_t, sr_t, self.loss_fn_alex)

            # append to list
            list_psnr.append(float(psnr_num))
            list_ssim.append(float(ssim_num))
            list_lpips.append(float(lpips_num))  # lpips_num is already a scalar
            list_ssim_gray.append(float(ssim_gray_num))
            
        return {
            'list_psnr': list_psnr,
            'list_ssim': list_ssim,
            'list_lpips': list_lpips,
            'list_ssim_gray': list_ssim_gray
        }

    def _test_epoch_end(self, stage):
        def rgb(t, write=False): 
            img = (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)
            if write and stage == 'test':
                Image.fromarray(img).save(path_fake[i])
            return img
        
        def imread(path):
            # libpng error: Read Error
            img = Image.open(path)
            img = np.asanyarray(img)
            img = img[:, :, [2, 1, 0]]
            return img

        # load pred results, load gt using hparams
        path_real = sorted(glob(os.path.join(self.hparams.test_folder, 'high' , '*')))
        path_fake = sorted(glob(os.path.join(self.hparams.results_folder , '*')))


        list_psnr = []
        list_ssim = []
        list_lpips = []
        list_ssim_gray = []

        for i in range(len(path_real)):
            hr = imread(path_real[i])
            sr = imread(path_fake[i])

            if hr.shape != sr.shape:
                from compare.full_ref import center_crop
                hr = center_crop(hr, sr)

            sr_t = torch.from_numpy(sr).float() / 255
            sr_t = rearrange(sr_t, 'h w c -> 1 c h w').contiguous()
            
            # metric calculation
            from compare.full_ref import _psnr, _ssim, _lpips, _ssim_gray, transform
            psnr_num = _psnr(hr, sr)
            ssim_num = _ssim(hr, sr)
            ssim_gray_num = _ssim_gray(hr, sr)

            sr_t = transform(sr).to(self.device)
            hr_t = transform(hr).to(self.device)
            lpips_num = _lpips(hr_t, sr_t, self.loss_fn_alex)

            # append to list
            list_psnr.append(psnr_num)
            list_ssim.append(ssim_num)
            list_lpips.append(lpips_num)
            list_ssim_gray.append(ssim_gray_num)

        psnr_score = np.mean(list_psnr)
        ssim_score = np.mean(list_ssim)
        ssim_gray_score = np.mean(list_ssim_gray)
        lpips_score = np.mean(list_lpips)
        

        if stage == 'valid':
            res = {
                f'{stage}/psnr':  np.around(np.mean(list_psnr), 3),
                f'{stage}/ssim': np.around(np.mean(list_ssim), 3),
                f'{stage}/ssim_gray': np.around(np.mean(list_ssim_gray), 3),
                f'{stage}/lpips': np.around(np.mean(list_lpips), 3),
                f'{stage}/combined': ((1-lpips_score)*0.8 + ssim_score * \
            0.1 + psnr_score*0.1), # for validation
                } 
        elif stage == 'test' or stage == 'predict':
            res = {
                f'{stage}/psnr':  np.around(np.mean(list_psnr), 3),
                f'{stage}/ssim': np.around(np.mean(list_ssim), 3),
                f'{stage}/ssim_gray': np.around(np.mean(list_ssim_gray), 3),
                f'{stage}/lpips': np.around(np.mean(list_lpips), 3),
            }

        self.log_dict(res, True, True, sync_dist=True) # dont know, but i think 
        self.trainer.strategy.barrier()

    def test_epoch_end(self, test_step_outputs) -> None:
        self._test_epoch_end('test')    
        
    def _test_step(self, batch, batch_idxs, stage):
        img_lr, img_hr, img_lr_name = batch # img_hr domain [0,1], img_lr [0,1]
        
        original_shape = img_lr.shape
        if self.hparams.padding:
            img_lr = pad_to_multiple(img_lr)

        img_lr_name = list(img_lr_name)
        imgs = self.sample(img_lr, self.hparams.return_all_timesteps) 

        if self.hparams.padding:
            imgs = unpad_from_multiple(imgs, original_shape)

        for i in range(imgs.shape[0]):
            img = imgs[i].cpu()
            img = img.detach().numpy()
            img = np.transpose(img, (1, 2, 0)) * 255
            img = img.astype(np.uint8) # Convert to uint8
            # dont know why cv2.imwrite will tune image to blue
            img = Image.fromarray(img)
            img.save(os.path.join(self.hparams.results_folder, img_lr_name[i])) 
    
    def test_step(self, batch, batch_idxs):
        if self.trainer.strategy.is_global_zero:
            return self._test_step(batch, batch_idxs, 'test')

    def forward(self, img, yield_intermediate=False):
        #img = {img_path} or PIL Image
        # img_in is a PIL image
        from dataset import get_patch, load_img
        from einops import rearrange
        from torchvision.transforms.functional import to_tensor
        from torch import device
        
        if isinstance(img, str):
            img_in = load_img(img)
        elif isinstance(img, Image.Image):
            # If it's already a PIL Image, use it directly
            img_in = img

        if not self.hparams.padding:
            img_lr, img_tar = get_patch(img_in, img_in, -1, False, False)
        else:
            img_lr = img_in

        img_lr = to_tensor(img_lr)
        img_lr = rearrange(img_lr, 'c h w -> 1 c h w')
        
        if torch.cuda.is_available():
            img_lr = img_lr.to(device('cuda'))
            self.to(device('cuda'))

        with torch.no_grad():
            original_shape = img_lr.shape

            if self.hparams.padding:
                img_lr = pad_to_multiple(img_lr)

            img = self.sample(img_lr, yield_intermediate) 

            if self.hparams.padding:
                img = unpad_from_multiple(img, original_shape)

        return img
        
if __name__ == '__main__':
    betas = cosine_beta_schedule(1000, s=0.008)
    alphas = 1 - betas 
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    lam = sqrt_alphas_cumprod / np.sqrt(1 - alphas_cumprod)
    log_lam = np.log(lam)
    print()