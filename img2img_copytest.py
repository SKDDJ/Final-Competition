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
import libs.autoencoder
import argparse
# from configs.sample_config import get_config
from pathlib import Path
import clip
from labml import lab, monit
from labml_nn.diffusion.stable_diffusion.sampler.ddim import DDIMSampler
from labml_nn.diffusion.stable_diffusion.util import load_model, load_img, set_seed
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
                #  autoencoder=autoencoder    
                 ):


        orig = latent_cb
        
        orig_clipimg = torch.randn(1, 1, 512, device=self.device)

        # autoencoder = autoencoder.to(self.device)
        
        # Get the number of steps to diffuse the original
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        if strength * self.ddim_steps < 1.0:
            t_index = int(1) 
        else:
            t_index = int(strength * self.ddim_steps) # int 37 

        # AMP auto casting
        with torch.cuda.amp.autocast():
            # un_cond = self.model.get_text_conditioning(batch_size * [""])

            
            # Get the prompt embeddings
            cond = context
            ### now cond is torch.Size([2, 77, 768])
            
            ### cond.shape: torch.Size([4, 77, 768])
            # cond = self.model.get_encode_prefix(cond)
                    
            def captiondecodeprefix(x):
                return self.model.get_decode_prefix(x)
            
            def captionencodeprefix(x):
                return self.model.get_encode_prefix(x)
            
            # Add noise to the original image
            
            t_img = torch.Tensor([1]).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            t_text = torch.zeros(t_img.size(0), dtype=torch.int, device=self.device)
            datatype = torch.zeros_like(t_text, device=self.device, dtype=torch.int) + 1
            
            
            # x,added_noise = self.sampler.q_sample(orig, t_index)
            x,orgin_n = self.sampler.q_sample(orig, t_index)
            
            # Reconstruct from the noisy image
            x = self.sampler.paint(x, 
                                   cond, 
                                   t_index,
                                   t_img, 
                                   orig_clipimg,
                                   t_text,
                                   datatype,
                                   captiondecodeprefix,
                                   captionencodeprefix,
                                   orgin_n,
                                   uncond_scale=uncond_scale
                                )
            images = x
            
        return images
