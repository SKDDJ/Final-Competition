"""
---
title: Generate images using stable diffusion with a prompt from a given image
summary: >
 Generate images using stable diffusion with a prompt from a given image
---

# Generate images using [stable diffusion](../index.html) with a prompt from a given image
"""

import torch

import argparse
from torch import nn
import einops
from resize import resize_images_in_path

import argparse
from pathlib import Path
import clip
from labml import lab, monit
from labml_nn.diffusion.stable_diffusion.sampler.ddim import DDIMSampler
from labml_nn.diffusion.stable_diffusion.util import load_model, load_img, save_images, set_seed
# from torchvision.utils import save_image

class Img2Img:
    """
    ### Image to image class
    """

    def __init__(self, *, config,
                 ddim_steps: int = 50,
                 ddim_eta: float = 0.0):
        """
        :param checkpoint_path: is the path of the checkpoint
        :param ddim_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        self.ddim_steps = ddim_steps

        # Load [latent diffusion model](../latent_diffusion.html)
        self.model = load_model(config.uvit ,config)  

        # Get device
        self.device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
        # Move the model to device
        self.model.to(self.device)

        # Initialize [DDIM sampler](../sampler/ddim.html)
        self.sampler = DDIMSampler(self.model,
                                   n_steps=ddim_steps,
                                   ddim_eta=ddim_eta)
        

    @torch.no_grad()
    def __call__(self, context,latent_cb: torch.tensor, *,
                 strength: float = 0.0,
                 batch_size: int = 1,
                 prompt: str = "",
                 uncond_scale: float = 5.0,
                 
                 ):
        """
        :param dest_path: is the path to store the generated images
        :param orig_img: is the image to transform
        :param strength: specifies how much of the original image should not be preserved
        :param batch_size: is the number of images to generate in a batch
        :param prompt: is the prompt to generate images with
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        """

        orig = latent_cb
        
        ### orig: torch.Size([4, 4, 80, 60])

        orig_clipimg = torch.randn(1, 1, 512, device=self.device)

        
        # Get the number of steps to diffuse the original
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_index = int(strength * self.ddim_steps) # int 37 

        # AMP auto casting
        with torch.cuda.amp.autocast():
            un_cond = self.model.get_text_conditioning(batch_size * [""])

             
            # Get the prompt embeddings
            cond = context
            
            ### cond.shape: torch.Size([4, 77, 768])
            cond = self.model.get_encode_prefix(cond)
            
            def captiondecodeprefix(x):
                return self.model.get_decode_prefix(x)
            
            def captionencodeprefix(x):
                return self.model.get_encode_prefix(x)
            
            # Add noise to the original image
            
            t_img = torch.Tensor(t_index).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            t_text = torch.zeros(t_img.size(0), dtype=torch.int, device=self.device)
            datatype = torch.zeros_like(t_text, device=self.device, dtype=torch.int) + 1
            
            
            x,added_noise = self.sampler.q_sample(orig, t_index)
            
            # Reconstruct from the noisy image
            x = self.sampler.paint(x, cond, t_index,t_img, orig_clipimg, t_text, datatype, captiondecodeprefix,captionencodeprefix,
                                   uncond_scale=uncond_scale,
                                   uncond_cond=un_cond)
            images = x
            
        return images


def main():
    """
    ### CLI
    """
    from configs.sample_config import get_config
    
    config = get_config()

    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to render"
    )

    parser.add_argument(
        "--orig-img",
        type=str,
        nargs="?",
        default="/home/schengwei/Competitionrepo/resources/boy1_example.jpeg",
        help="path to the input image"
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="cuda:5",
        help="device to use"
    )
    # init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size", )
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps")

    parser.add_argument("--scale", type=float, default=5.0,
                        help="unconditional guidance scale: "
                             "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

    parser.add_argument("--strength", type=float, default=0.01,
                        help="strength for noise: "
                             "vary from 0.0 to 1.0 which 1.0 corresponds to full destruction of information in init image")

    opt = parser.parse_args()
    set_seed(42)
    
    # which gpu to use 
    config.device = opt.device_id
    
    img2img = Img2Img(config=config, ddim_steps=opt.steps, ddim_eta=0)


    with monit.section('Generate'):
        img2img(
            dest_path='/home/schengwei/Competitionrepo/ddimoutput',
            orig_img=opt.orig_img,
            strength=opt.strength,
            batch_size=opt.batch_size,
            prompt=opt.prompt,
            uncond_scale=opt.scale)


#
if __name__ == "__main__":
    main()