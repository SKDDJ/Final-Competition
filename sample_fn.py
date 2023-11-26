"""
采样代码
文件输入:
    prompt, 指定的输入文件夹路径, 制定的输出文件夹路径
文件输出:
    采样的图片, 存放于指定输出文件夹路径
- 指定prompt文件夹位置, 选手需要自行指定模型的地址以及其他微调参数的加载方式, 进行图片的生成并保存到指定地址, 此部分代码选手可以修改。
- 输入文件夹的内容组织方式和训练代码的输出一致
- sample的方法可以修改
- 生成过程中prompt可以修改, 但是指标测评时会按照给定的prompt进行测评。
"""

import os
import ml_collections
import torch
import random
import argparse
import utils
from libs.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import einops

import libs.clip
from torchvision.utils import save_image, make_grid
import numpy as np
import clip
import time
import numpy as np
import json


from img2img_copytest import Img2Img
from configs.unidiffuserv1 import get_config
from labml import monit



import json
import os
import time
from PIL import Image
import argparse
import numpy as np
from rembg import remove, new_session
from labml import monit
from labml_nn.diffusion.stable_diffusion.util import load_img_rm
import torch 
from img2img_copytest import Img2Img
from configs.unidiffuserv1 import get_config
from load_model import xiugo

from PIL import Image

img2img = None
class Timetest:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timetest, cls).__new__(cls)
            cls._instance.total_wait_time = 0
        return cls._instance

    def time_test(self,config,task_id):
        count = count_files(config.accelerate_adapters, task_id)  # 更新count
        while count not in [1, 2, 3]:
            time.sleep(0.5)
            self.total_wait_time += 0.5
            if self.total_wait_time > 60:
                raise TimeoutError("Total wait time exceeded 1 minute. Please contact xiugo team.")
            count = count_files(config.accelerate_adapters, task_id)  # 再次更新count以便检查循环条件
            
def init_img2img():
    global img2img
    config = get_config()
    img2img = Img2Img(config=config, ddim_steps=60, ddim_eta=0)
    return img2img 

def get_img2img():
    global img2img
    if img2img is None:
        init_img2img()
    return img2img 

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def prepare_contexts(config, clip_text_model):
    device = config.device

    contexts = torch.randn(config.n_samples, 77, config.clip_text_dim).to(device)
    img_contexts = torch.randn(config.n_samples, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2])
    clip_imgs = torch.randn(config.n_samples, 1, config.clip_img_dim)
 
    prompts = [ config.prompt ] * config.n_samples
    contexts = clip_text_model(prompts)

    return contexts, img_contexts, clip_imgs



def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


# @torch.cuda.amp.autocast()
def get_group2(adapterpath,context,task_id):
    img2img = get_img2img()
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        
        file_name = os.path.basename(adapterpath)
        extracted_id = file_name.split('-')[0]
        # print(extracted_id)
        # print(task_id)
        if int(extracted_id) != int(task_id):
            raise ValueError("PLEASE Contact xiugo team!!!!!!!PLEASE")
        # input_path = adapterpath[(task_id+np.random.randint(0,6)) % len(adapterpath)]
        # input_path = os.path.join('final_json_data',input_path)
        latent_cb = torch.load(adapterpath).to(context.device)

        img_inversion = img2img(
            context=context,
            latent_cb = latent_cb
            )
        # img_inversion = latent_cb
    return img_inversion


def count_files(folder_path, start_number):
    """
    Count the number of files in the given folder that start with the specified number.

    :param folder_path: Path to the folder containing the files.
    :param start_number: The starting number to match the filenames.
    :return: The count of files starting with the given number.
    """
    count = 0
    prefix = f"{start_number}-"
    
    # List all files in the given directory
    for filename in os.listdir(folder_path):
        # print(filename)
        # Check if the file starts with the specified number
        if filename.startswith(prefix) and filename.endswith(".pt"):
            count += 1
    # print(f"{start_number}",count)
    return count

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def sample( config, nnet, clip_text_model,nnet_standard , autoencoder, caption_decoder, device, task_id, output_path):
    """
    using_prompt: if use prompt as file name
    """
    n_iter = config.n_iter

    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    config = ml_collections.FrozenConfigDict(config)
    
    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)


    use_caption_decoder = config.text_dim < config.clip_text_dim or config.mode != 't2i'

    # empty_context = clip_text_model([''])[0]

    def split(x):
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        return z, clip_img

    def combine(z, clip_img):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        return torch.concat([z, clip_img], dim=-1)


    def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
            config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
            config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
        3. return linear combination of conditional output and unconditional output
        """
        z, clip_img = split(x)
       
        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        
        # 假设 config.use_nnet_standard 是一个布尔值，决定是否使用 nnet_standard
        use_nnet_standard = config.use_nnet_standard

        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        x_out = combine(z_out, clip_img_out)
        
        
        text_N = torch.randn_like(text)  # 3 other possible choices
        z_out_uncond, clip_img_out_uncond, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        
        if use_nnet_standard:
            z_out_standard, clip_img_out_standard, text_out_standard = nnet_standard(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                                         data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_standard = combine(z_out_standard, clip_img_out_standard)
            
            
            z_out_uncond_standard, clip_img_out_uncond_standard, text_out_uncond_standard = nnet_standard(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond_standard = combine(z_out_uncond_standard, clip_img_out_uncond_standard)
            
            # 根据 config.sample.scale 返回不同的结果
            if config.sample.scale == 0.:
                return x_out
            else:
                return x_out + config.sample.scale * (x_out_standard - x_out_uncond_standard)
        else:
            return x_out + config.sample.scale * (x_out - x_out_uncond)

        # z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
        #                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        # x_out = combine(z_out, clip_img_out)
        # z_out_standard, clip_img_out_standard, text_out_standard = nnet_standard(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
        #                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        # x_out_standard = combine(z_out_standard, clip_img_out_standard)
        # if config.sample.scale == 0.:
        #     return x_out

        # text_N = torch.randn_like(text)  # 3 other possible choices
        # z_out_uncond_standard, clip_img_out_uncond_standard, text_out_uncond_standard = nnet_standard(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
        #                                                           data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        # z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
        #                                                           data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        # # x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        # x_out_uncond_standard = combine(z_out_uncond_standard, clip_img_out_uncond_standard)
            

        # # return x_out + config.sample.scale * (x_out - x_out_uncond)
        # return x_out + config.sample.scale * (x_out_standard - x_out_uncond_standard)


    contexts, img_contexts, clip_imgs = prepare_contexts(config, clip_text_model)
    rm_contexts = contexts
    contexts_low_dim = contexts if not use_caption_decoder else caption_decoder.encode_prefix(contexts)  # the low dimensional version of the contexts, which is the input to the nnet
    # print(contexts_low_dim.shape)
    # exit()
    _n_samples = contexts_low_dim.size(0)


    def sample_fn(**kwargs):
        
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)
        _x_init = combine(_z_init, _clip_img_init)

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return t2i_nnet(x, t, **kwargs)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        # print(config.sample.sample_steps)
        with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"), monit.section('Sample:'):
            # x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
            x = dpm_solver.sample(_x_init, steps=10, eps=1. / N, T=1.)

        _z, _clip_img = split(x)
        return _z, _clip_img
    if not os.path.exists("other_models/adapter"):
            os.makedirs("other_models/adapter")
    test_instance = Timetest()
    samples = None    
    for i in range(n_iter):
        _z, _clip_img = sample_fn(text=contexts_low_dim)  # conditioned on the text embedding
        # print(_z)
        # print(_z.shape)
        new_samples = unpreprocess(autoencoder.decode(_z))
        # device = _z.device
        # new_samples = unpreprocess(decoder_consistency(_z//0.18215, device))
        
        
        if samples is None:
            samples = new_samples
        else:
            samples = torch.vstack((samples, new_samples))
    error_count = 0
    

    for i in range(4 - config.n_samples):
        # image_paths = config.image_paths
        """
        ## input args : 1. encode 过的原图，需要原图路径，并且使用 encode 函数进行 encode
        ## 2. prompts 上面的 contexts 
        # contexts,_ = torch.chunk(contexts,chunks=2,dim=0)
        ### print(contexts.shape)
        ## torch.Size([2, 77, 768])
        # new_z = get_group2(image_paths,contexts,task_id)
        # print(contexts_low_dim.shape)
        # print(rm_contexts.shape)        
        ### torch.Size([2, 77, 64])
        """
        # torch.Size([2, 77, 768])
        source_tensor = torch.empty(1, 77, 64)
        rm_contexts = torch.randn_like(source_tensor).to(contexts.device)
        # rm_contexts = torch.zeros_like(source_tensor).to(contexts.device)
        
        # task_id - rand().pt as input (from 1 - 3 )
        # count = count_files(config.accelerate_adapters,task_id)
  
        test_instance.time_test(config,task_id)
        print(test_instance.total_wait_time)
        count = count_files(config.accelerate_adapters, task_id)
        if count == 1:
            # print("skdljfklsdjafkljdsakfj")
            random_integer = 1
        else:
            random_integer = np.random.randint(1, count+1)
            
            
        adapterpath = config.accelerate_adapters
        
        adapterpath = os.path.join(adapterpath,f"{task_id}-{random_integer}.pt")
        ## other_models/adapters/...pt
        new_z = get_group2(adapterpath,rm_contexts,task_id)
        device = new_z.device
        with monit.section('autoencoder_decode:'):
            new_samples = unpreprocess(autoencoder.decode(new_z))

            # new_samples = unpreprocess(decoder_consistency(new_z//0.18215, device))
        samples = torch.vstack((samples, new_samples))


    
    paths = []

    for idx, sample in enumerate(samples):
        save_path = os.path.join(output_path, f'{task_id}-{config.prompt}-{idx:03}.jpg')
        paths.append(save_path)
        # with monit.section(f'Save Image {task_id}:'):
        save_image(sample, save_path)
        
    return paths

