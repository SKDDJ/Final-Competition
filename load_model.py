
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

import multiprocessing
import torch
try:
   multiprocessing.set_start_method('spawn', force=True)
   print("spawned")
except RuntimeError:
   pass

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

TIME_ST = time.time()
TIME_ED = time.time()
TOTAL_TIME = 0
# export CUDA_VISIBLE_DEVICES=3

def xiugo(oimage, image):
    diff = torch.abs(oimage - image)
    config = get_config()
    device = config.device 
    
    diff.to(device)
    mask = diff < 4 # 10 is black
    rand = torch.randn_like(oimage)
    
    mask.to(device)
    ## 用掩码选择 t1 中的元素。这将返回一个新的张量，其中掩码为 True 的位置保持不变，其余位置为 0
    image = image * mask.float()
    rand[mask] = image[mask]
    return rand


def accelerate_model( config, folder_path,autoencoder, max_files=700):
    # print(autoencoder)
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    processed_files = 0
    device = config.device
    for json_file in json_files:
        if processed_files >= max_files:
            break
        with open(os.path.join(folder_path, json_file), 'r') as file:
            data = json.load(file)
        img_id = data["id"]
        paths = [item["path"] for item in data["source_group"]]
        num_images = len(paths)
        num_to_process = 3 if num_images > 2 else num_images
        for i in range(num_to_process):
            img = Image.open(paths[i])
            # print("img.mode:",img.mode)
            with monit.section("preloading:"):
                session = new_session('other_models/.u2net/u2net.onnx')
                img_rm = remove(img, session=session)
                # print("img_rm:",img_rm.mode)
                if img_rm.mode == 'RGBA':
                    img_rm = img_rm.convert('RGB')
                    # print("new_img_rm:",img_rm.mode)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                    
                img_rm_tensor = load_img_rm(img_rm).to(device)  # Assuming load_img_rm is defined
                img_tensor = load_img_rm(img).to(device)  # Assuming load_img_rm is defined
                # print("img_rm_tensor",img_rm_tensor.shape)
                # print("img_tensor",img_tensor.shape)
                latent_rm = autoencoder.encode(img_rm_tensor)
                latent_img = autoencoder.encode(img_tensor)
                latent_cb = xiugo(latent_img, latent_rm)
                output_path = config.accelerate_adapters
                ## other_models/adapter'
                if not os.path.exists(output_path):
                    os.makedirs(output_path, exist_ok=True)

                adapterpath = os.path.join(output_path, f"{img_id}-{i+1}.pt")

                # # import os
                # torch.save(latent_cb, adapterpath)

                try:
                    torch.save(latent_cb, adapterpath)
                except Exception as e:
                    print(f"Error saving file {adapterpath}: {e}")

            
        processed_files += 1

        

def prepare_context():
    """
    prepare context for later use
    """
    import torch
    # import libs.autoencoder
    import utils
    from utils import set_logger
    from absl import logging
    from libs.uvit_multi_post_ln_v1 import UViT
    from configs.unidiffuserv1 import get_config
    import builtins
    import ml_collections
    from labml import monit
    from torch import multiprocessing as mp
    from sample_fn import get_img2img
    from rembg import remove, new_session


        
    torch.set_num_threads(1)
    
#    /workspace/final_json_data
    config = get_config()

    device = config.device


    img2img = get_img2img()
    
    img2img = torch.compile(img2img, mode="reduce-overhead")

    
    config.prompt = ""
    nnet = img2img.model
    nnet_standard = img2img.model.nnet_standard
    clip_text_model = img2img.model.cond_stage_model
    autoencoder = img2img.model.autoencoder
    # decoder_consistency = img2img.model.decoder_consistency
    caption_decoder = img2img.model.caption_decoder
    clip_img_model = img2img.model.image_stage_model
    clip_img_model_preprocess = img2img.model.get_clipimg_embedding
    


    return {
        "device": device,
        'config': config,
        "origin_sd": nnet_standard,
        "caption_decoder": caption_decoder,
        "nnet": nnet,
        "autoencoder": autoencoder,
        # "decoder_consistency": decoder_consistency,
        "clip_text_model": clip_text_model,
        "clip_img_model": clip_img_model,
        "clip_img_model_preprocess": clip_img_model_preprocess
    }


def load_json_files(path):
    """
    given a directory, load all json files in that directory
    return a list of json objects
    """
    d_ls = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            with open(os.path.join(path, file), 'r') as f:
                json_data = json.load(f)
                d_ls.append(json_data)
    return d_ls


def process_one_json(json_data, image_output_path, context={}):
        # multiprocessing.set_start_method('spawn')

        ## other_models/adapters/xxx.pt
        ## img_id 5
     
        """
        given a json object, process the task the json describes
        """

        import utils
        from absl import logging
        import torch
        from sample_fn import sample
        
        # 初始化训练步数

        config = context["config"]
        device = context["device"]
        nnet_standard = context["origin_sd"]
        caption_decoder = context["caption_decoder"]
        nnet = context["nnet"]
        autoencoder = context["autoencoder"]
        # decoder_consistency = context["decoder_consistency"]
        clip_text_model = context["clip_text_model"]

            # 静态变量存储子进程引用
        if not hasattr(process_one_json, "accelerate_process"):
            process_one_json.accelerate_process = multiprocessing.Process(target=accelerate_model, args=(context["config"], context["config"].modelcontext, context["autoencoder"]))
            process_one_json.accelerate_process.start()

        config.n_samples = 2
        ######### 这里的才有用！！！！！！！！########
        config.n_iter = 1

        image_paths = [i["path"] for i in json_data["source_group"]]
        images = []
        for caption in json_data["caption_list"]:
            config.prompt = caption
            config.image_paths = image_paths
            paths = sample(config, nnet, clip_text_model, nnet_standard ,  autoencoder, caption_decoder, device, json_data["id"], output_path=image_output_path)
            images.append({"prompt": caption, "paths": paths, 
                            })


        return {
            "id": json_data["id"],
            "images": images
        }
    


def tik():
    global TIME_ST
    TIME_ST = time.time()
def tok(name):
    global TIME_ED
    TIME_ED = time.time()
    elapsed_time = TIME_ED - TIME_ST
    print(f"Time {name} elapsed: {elapsed_time}")    
    
def tik_tok():
    global TOTAL_TIME
    TOTAL_TIME = TOTAL_TIME + TIME_ED - TIME_ST



    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--json_data_path", type=str, default="final_json_data/json", help="gived json")
    # parser.add_argument("-j","--json_output_path", type=str, default="our_json_outputs", help="111111generated json output")
    # parser.add_argument("-i","--image_output_path", type=str, default="our_image_outputs", help="111111generated images output")

    # parser.add_argument("-j","--json_output_path", type=str, default="our_json_imageoforigin2", help="22222generated json output")
    # parser.add_argument("-i","--image_output_path", type=str, default="our_imageoforigin2", help="222222generated images output")
    
    # parser.add_argument("-j","--json_output_path", type=str, default="aaaaaaasaveresults/aaaajsons", help="22222generated json output")
    # parser.add_argument("-i","--image_output_path", type=str, default="aaaaaaasaveresults/aaaaimages", help="222222generated images output")
    
    parser.add_argument("-j","--json_output_path", type=str, default="aaaaaaasaveresults/bbbbjsons", help="33333generated json output")
    parser.add_argument("-i","--image_output_path", type=str, default="aaaaaaasaveresults/bbbbimages", help="333333generated images output")
    
    
    
    # parser.add_argument("-c","--cuda_device", type=str, default="cuda:7", help="CUDA device to use (e.g., 0, 1, 2, ...)")
    return parser.parse_args()

