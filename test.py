import argparse
import yaml
from cond import Unet_cond
from diffusion_fromscratch import LitDiffusion, EnlightDiffusion
from model import Unet
from dataset import LitLOLDataModule
import pytorch_lightning as pl
import warnings

warnings.filterwarnings("ignore")

def test(config):
    # seed
    pl.seed_everything(seed=config.seed, workers=True)

    # model
    unet_cond = Unet_cond(config, config.cond_in_dim, True)
    unet = Unet(config, in_dim=config.in_dim)
    diffusion = EnlightDiffusion(unet, config)

    assert config.diffusion_path != '', "diffusion.path must be a valid path"
    litmodel = LitDiffusion.load_from_checkpoint(
            config.diffusion_path, diffusion_model=diffusion, encoder=unet_cond, config=config, strict=False)

    trainer = pl.Trainer.from_argparse_args(
        config,
        logger=True,
    )

    litdataModule = LitLOLDataModule(config, [''], [config.test_folder])
    litdataModule.setup(stage=None)

    trainer.test(model=litmodel,
                         datamodule=litdataModule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", default='cfg/test/FS/test.yaml')
    config = parser.parse_args()

    with open(config.cfg, "r") as infile:
        cfg = yaml.full_load(infile)

    config = argparse.Namespace(**cfg)

    test(config)