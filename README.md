# AnlightenDiff: Anchoring Diffusion Probabilistic Model on Low Light Image Enhancement
by C.-Y. Chan, W.-C. Siu, Y.-H. Chan and H. A Chan

#TODO: Add accepted Journal Name on top

<img width="988" alt="image" src="https://github.com/user-attachments/assets/344f3286-dec2-4ff1-b9ec-fcb3c780bd03">


## BibTeX
Wait IEEE Explore bibtex

## Paper is TL;DR 
### Abstract
Low-light image enhancement aims to improve the visual quality of images captured under poor illumination. However, enhancing low-light images often introduces image artifacts, color bias, and low SNR. In this work, we propose AnlightenDiff, an anchoring diffusion model for low light image enhancement. Diffusion models can enhance the low light image to well-exposed image by iterative refinement, but require anchoring to ensure that enhanced results remain faithful to the input. We propose a Dynamical Regulated Diffusion Anchoring mechanism and Sampler to anchor the enhancement process. We also propose a Diffusion Feature Perceptual Loss tailored for diffusion based model to utilize different loss functions in image domain. AnlightenDiff demonstrates the effect of diffusion models for low-light enhancement and achieving high perceptual quality results. Our techniques show a promising future direction for applying diffusion models to image enhancement.

<img width="483" alt="image" src="https://github.com/user-attachments/assets/02cd4b58-b3be-4383-88cc-0d2818fb54c6">

### Dynamical Regulated Diffusion Anchoring (DRDA)
We utilize a Dynamical Regulated Diffusion Anchoring (DRDA) mechanism to dynamically regulate the mean vector of the perturbations φ to incorporate domain knowledge and match the geometry of the data distribution to explore more complex target distributions, which provide larger flexibility for diffusion-based models.

```math
  \boldsymbol{x}_t = \sqrt{\bar{\alpha}_t} \boldsymbol{x}_{0} + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t^\star \label{eq:DRDA}; \boldsymbol{\epsilon}_t^\star \sim \mathcal{N}(\boldsymbol{m}_t, \tilde{\beta}_t \boldsymbol{I})
```
(Eqn 12 and 13 in paper)

The related code of DRDA (which introduce new center in diffusion forward process) is shown below:
```
def q_sample(self, x_start, t, center, noise=None):
    """
    Samples from q(x_t | x_0) adding noise to the original image.
    x_start : the original image
    t : sampled timestep
    center : the perturbation of mean (DRDA)
    """
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

```

### Dynamical Regulated Diffusion Sampler (DRDS)
We propose Dynamical Regulated Diffusion Sampler (DRDS), which builds upon the reverse process of diffusion models and dynamically regulates the diffusion process to explore the target distribution. This models more complex distributions compared to existing diffusion- based approaches and enables more efficient exploration of the empirical distribution and thus results in higher-quality sample generation.

```math
  \boldsymbol{x}_{t-1}=\mathcal{N}\left(\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha_t})}\boldsymbol{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(1-\bar{\alpha_t})\sqrt{\alpha_t}}\left(\boldsymbol{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t,  \boldsymbol{x}_H)\right) + \frac{1 - \bar{\alpha}_{t} +  \sqrt{\bar{\alpha}_{t-1}}(\alpha_t  - 1)  + \sqrt{\alpha_t}(\bar{\alpha}_{t-1} - 1) }{1-\bar{\alpha}_t}\boldsymbol{\phi} ,\boldsymbol{I}\right) 
```

(Eqn 17, 18, and 19 in paper)

The related code of DRDS (which introduce new center in diffusion reverse process) is shown below:
```
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

    posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped
```

### Diffusion Feature Perceptual Loss (DFPL)
We propose the Diffusion Feature Perceptual Loss (DFPL), which is a loss function tailored for diffusion models. DFPL leverages the predicted noise perturbation to reconstruct the predicted noisy images xθt and compares them with the ground truth noisy images xt. This approach allows the use of image-based loss functions and provides image-level supervision, resulting in improved visual quality in generation.

```math
  \mathcal{L}_{\text{DFPL}}(\boldsymbol{x}_0, \boldsymbol{\epsilon}_t, \boldsymbol{\epsilon}_t^{\theta}) = \mathcal{L}_{\text{Image}} \left( \sqrt{\bar{\alpha}_t} \boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t, \\ 
  \sqrt{\bar{\alpha}_t} \boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t^{\theta} \right)
```
(As eqn 21 in the paper)

The related code of DFPL (which introduce image level loss for diffusion training) is shown below:
```
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
```
## Environment Setup
### Requirements
To install the dependencies, run the following command:

```
git clone https://github.com/allanchan339/AnlightenDiff
conda env create --name AnlightenDiff --file=environment.yaml
conda activate AnlightenDiff
```

### Dataset
We use LOL, VELOL and LOL-v2 datasets for training and testing. You can download the datasets from the following links:

1. [LOL](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view?usp=sharing)

2. [VELOL](https://www.dropbox.com/s/vfft7a8d370gnh7/VE-LOL-L.zip?dl=0)

3. [LOL-v2](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing)

and put them in the folder `data/` with the following hararchy (note that some of the folder may need renaming):

```
data
    - LOL
        - eval15
        - our485

    - LOLv2
        - Real_captured
            - Test
            - Train

    - VE-LOL-L
        - VE-LOL-L-Cap-Full
            - test
            - train 
```
## Testing 
### Pretrained Models
Trained model can be downloaded from [here](https://1drv.ms/u/s!AvJJYu8Th24UjKJp4oGJqDoYNlOiKQ?e=h5qwHO)


### Evaluation 
1. test on LOL dataset
```
    python test.py
```

2. test on VELOL dataset
```
    python test.py --cfg cfg/test/FS/test_VELOL.yaml
```

3. test on LOL-v2 dataset
```
    python test.py --cfg cfg/test/FS/test_LOLv2.yaml
```

### Results
<img width="1016" alt="image" src="https://github.com/user-attachments/assets/919f85c6-23fa-4e80-a2d6-3297a4bc52de">


### Inference on Custom Images (Unpaired)
To test on custom/unpaired images, the cfg file `test_unpaired.yaml` in folder `cfg/test/` should be modified as follows:
```
test_folder_unpaired: 'WHERE THE INPUT FOLDER IS' # e.g., 'my_image_folder'
results_folder_unpaired: 'WHERE THE OUTPUT FOLDER IS' # e.g., 'my_image_results'
```

Then, run the following command:
```
python test_unpaired.py --cfg cfg/test/test_unpaired.yaml
```

## Training
To train the model, run the following command:
```
python train.py --cfg cfg/train/FS/train.yaml
```
## Contact
Thanks for looking into this repo! If you have any suggestion or question, feel free to leave a message here or contact me via cy3chan@sfu.edu.hk.
