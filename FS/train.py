from model import Unet
import argparse
import pytorch_lightning as pl
from dataset import LitLOLDataModule
from diffusion_fromscratch import LitDiffusion, EnlightDiffusion
from pytorch_lightning.strategies import DDPStrategy
from cond import Unet_cond
import yaml
from utils.gpuoption import gpuoption

def train(config):
    # to fix 4090 NCCL P2P bug in driver
    if gpuoption():
        print('NCCL P2P is configured to disabled, new driver should fix this bug')

    if config.use_dataset == 'LOL':
        train_folders = [config.train_folders_v1]
    elif config.use_dataset == "LOLv2":
        train_folders = [config.train_folders_v2]
    elif config.use_dataset == "LOL4K":
        train_folders = [config.train_folders_4k]

    elif config.use_dataset == 'LOL+LOLv2':
        train_folders = [config.train_folders_v1, config.train_folders_v2]
    elif config.use_dataset == 'LOL+LOLv2+VELOL':
        train_folders = [config.train_folders_v1,
                         config.train_folders_v2, config.train_folders_VE]
    else:
        NotImplementedError("dataset not supported")
    test_folder = config.test_folder

    # seed
    pl.seed_everything(seed=config.seed, workers=True)

    strategy = 'auto'

    # dataset
    litdataModule = LitLOLDataModule(config, train_folders, [test_folder])
    litdataModule.setup()

    # model
    unet_cond = Unet_cond(config, config.cond_in_dim, True)
    unet = Unet(config, in_dim=config.in_dim)
    diffusion = EnlightDiffusion(unet, config)

    if config.diffusion_path != '':
        litmodel = LitDiffusion.load_from_checkpoint(
            config.diffusion_path, diffusion_model=diffusion, encoder=unet_cond, config=config, strict=False)
    else:
        litmodel = LitDiffusion(diffusion, encoder=unet_cond, config=config)

    # trainer
    trainer = pl.Trainer.from_argparse_args(
        config,
        logger=True,
        strategy=strategy
        )
    # train
    trainer.fit(model=litmodel, datamodule=litdataModule)

    # test
    ckpt_path = trainer.checkpoint_callback.best_model_path
    litmodel = LitDiffusion.load_from_checkpoint(
                    ckpt_path, diffusion_model=diffusion, encoder=unet_cond, config=config, strict=False)    
    # new trainer
    trainer = pl.Trainer(
                    accelerator=config.accelerator,
                    devices=[config.devices[0]] if isinstance(config.devices, list) else config.devices,
                    strategy=None,
                )
    trainer.test(litmodel, datamodule=litdataModule)

if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='cfg/train/FS/train.yaml')
    config = parser.parse_args()

    with open(config.cfg, "r") as infile:
        cfg = yaml.full_load(infile)

    config = argparse.Namespace(**cfg)

    train(config)