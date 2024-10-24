import gradio as gr
import argparse
import yaml
from cond import Unet_cond
import pytorch_lightning as pl
from model import Unet
from dataset import is_image_file
import torch
import numpy as np
from PIL import Image
import os
import warnings
from cond import Unet_cond

warnings.filterwarnings("ignore")

def load_default_images(directories):
    image_paths = []
    for dir_name, dir_path in directories.items():
        image_files = [f for f in os.listdir(dir_path) if is_image_file(f)]
        for f in image_files:
            image_paths.append((os.path.join(dir_path, f), f"{dir_name}: {f}"))
    return image_paths

def load_config(cfg_path):
    # Load configuration
    with open(cfg_path, "r") as infile:
        cfg = yaml.full_load(infile)
    return argparse.Namespace(**cfg)

def initialize_model(config):
    from diffusion_fromscratch import LitDiffusion, EnlightDiffusion

    pl.seed_everything(seed=config.seed, workers=True)

    # model
    unet_cond = Unet_cond(config, config.cond_in_dim, True)
    unet = Unet(config, in_dim=config.in_dim)
    diffusion = EnlightDiffusion(unet, config)

    assert config.diffusion_path != '', "diffusion.path must be a valid path"
    litmodel = LitDiffusion.load_from_checkpoint(
            config.diffusion_path, diffusion_model=diffusion, encoder=unet_cond, config=config, strict=False)

    return litmodel


def main(args):
    config = load_config(args.cfg)
    litmodel = initialize_model(config)

    def process_image(input_image):
        # Convert to PIL Image if it's not already
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image.astype('uint8'), 'RGB')

            #TODO: implement intermediate steps that monitor VRAM usage

        yield_intermediate = True
        total_steps = config.timesteps
        import time
        with torch.no_grad():
            if yield_intermediate:
                for i, output_tensor in enumerate(litmodel(input_image, yield_intermediate=yield_intermediate)):

                    output_image = output_tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255
                    output_image = output_image.astype(np.uint8)
                    progress = (i + 1) / total_steps
                    yield output_image, progress * 100
                    time.sleep(0.5) # wait gradio to offload image from gpu to user 
    
    def process_image_final(input_image):
        # Convert to PIL Image if it's not already
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image.astype('uint8'), 'RGB')
            
        with torch.no_grad():
            output_tensor = litmodel(input_image, yield_intermediate=False)

            output_image = output_tensor[0].cpu().permute(1, 2, 0).numpy() * 255
            output_image = output_image.astype(np.uint8)
            return output_image, 100  # Final image with 100% progress

    # Define Gradio interface
    # iface = gr.Interface(
    #     fn=process_image,
    #     inputs=gr.Image(type='numpy'),
    #     outputs=gr.Image(),
    #     title="Image Enhancement Model from AnlightenDiff",
    #     description="Upload an image to enhance it using the trained model.",
    #     # live=True,
    # )

    # Load default images
    default_image_dirs = {
        # "Dark": "AnlightenDiff/dark",
        # "MEF": "AnlightenDiff/MEF",
        # "VV": "AnlightenDiff/VV"
    }
    default_image_paths = load_default_images(default_image_dirs)

    # Define Gradio interface using Blocks
    with gr.Blocks() as iface:
        gr.Markdown("# Image Enhancement Model from AnlightenDiff")
        gr.Markdown("Upload an image to enhance it using the trained model.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type='numpy', label="Input Image")
                yield_intermediate = gr.Checkbox(label="Show intermediate steps (Beta)", value=False)

                submit_btn = gr.Button("Enhance Image")
                # default_image_gallery = gr.Gallery(
                #     value=[path for path, _ in default_image_paths],
                #     label="Default Images",
                #     elem_id="gallery",
                #     show_label=False,
                #     columns=4,
                #     height="auto"
                # )            
            with gr.Column(scale=1):
                output_image = gr.Image(label="Enhanced Image")
                progress_bar = gr.Slider(label="Progress", value=0, minimum=0, maximum=100, interactive=False)

        def load_selected_image(evt: gr.SelectData):
            selected_index = evt.index
            return default_image_paths[selected_index][0]

        def on_submit(image, yield_intermediate):
            if yield_intermediate:
                # yield from process_image(image)
                output, progress = process_image_final(image)
                yield output, progress

            else:
                output, progress = process_image_final(image)
                yield output, progress

        # # action from click
        # default_image_gallery.select(
        #     load_selected_image,
        #     None,
        #     input_image
        # )

        submit_btn.click(
            on_submit, 
            inputs=[input_image, yield_intermediate], 
            outputs=[output_image, progress_bar]
        )

    # Launch the interface
    iface.queue(max_size=2)
    iface.launch(share=True, max_threads=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", default='cfg/test/FS/test_unpaired.yaml')
    args = parser.parse_args()

    main(args)