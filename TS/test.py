import argparse
import yaml
from cond import Unet_cond, LitCond
from diffusion_twostep import LitDiffusion, EnlightDiffusion
from model import Unet
from dataset import LitLOLDataModule
import pytorch_lightning as pl
import warnings

warnings.filterwarnings("ignore")

def main(config):

    # seed
    pl.seed_everything(seed=config.seed, workers=True)
    
    unet_cond = Unet_cond(config, config.cond_in_dim, True)
    unet = Unet(config, in_dim=config.in_dim)  
    diffusion = EnlightDiffusion(unet, config)

    encoder = LitCond(unet_cond, config)

    assert config.diffusion_path !='', "diffusion.path must be a valid path"
    litmodel = LitDiffusion.load_from_checkpoint(
                config.diffusion_path, diffusion_model=diffusion, encoder=encoder, config=config, strict=False)
    
    trainer = pl.Trainer.from_argparse_args(
        config,
        logger= True,
    )

    litdataModule = LitLOLDataModule(config, [''], [config.test_folder])
    litdataModule.setup(stage=None)

    trainer.test(model=litmodel,
                        datamodule=litdataModule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", default='cfg/test/TS/test.yaml')
    config = parser.parse_args()

    with open(config.cfg, "r") as infile:
        cfg = yaml.full_load(infile)

    config = argparse.Namespace(**cfg)

    main(config)