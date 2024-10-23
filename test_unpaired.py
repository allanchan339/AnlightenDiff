# for evaluate LOL dataset on model
import argparse
import yaml
from cond import Unet_cond
from diffusion_fromscratch import LitDiffusion, EnlightDiffusion
from model import Unet
import pytorch_lightning as pl
import numpy as np
from PIL import Image
import os
from os.path import join
from os import listdir
from dataset import is_image_file
import warnings

warnings.filterwarnings("ignore")

def main(config):

    # seed
    pl.seed_everything(seed=config.seed, workers=True)
    
    # model
    unet_cond = Unet_cond(config, config.cond_in_dim, True)
    unet = Unet(config, in_dim=config.in_dim)
    diffusion = EnlightDiffusion(unet, config)

    assert config.diffusion_path != '', "diffusion.path must be a valid path"
    litmodel = LitDiffusion.load_from_checkpoint(
            config.diffusion_path, diffusion_model=diffusion, encoder=unet_cond, config=config, strict=False)

    # inference on unpaired images
    image_filenames = [join(config.test_folder_unpaired, x)
                           for x in listdir(config.test_folder_unpaired) if is_image_file(x)]

    # create folder if not exist
    if not os.path.exists(config.results_folder_unpaired):
        os.makedirs(config.results_folder_unpaired)

    for index in range(len(image_filenames)):
        litmodel.eval()
        img = litmodel(image_filenames[index])
        img_lr_name = os.path.basename(image_filenames[index])

        img = img[0].cpu()
        img = img.detach().numpy()
        img = np.transpose(img, (1, 2, 0)) * 255
        img = img.astype(np.uint8)  # Convert to uint8
        # dont know why cv2.imwrite will tune image to blue
        img = Image.fromarray(img)

        img.save(os.path.join(config.results_folder_unpaired, img_lr_name))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", default='cfg/test/FS/test_unpaired.yaml')
    config = parser.parse_args()

    with open(config.cfg, "r") as infile:
        cfg = yaml.full_load(infile)

    config = argparse.Namespace(**cfg)

    main(config)