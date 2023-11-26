import json
import os
import time
from PIL import Image
import argparse
import numpy as np
TIME_ST = time.time()
TIME_ED = time.time()


def prepare_context():
    """
    prepare context for later use
    """
    import torch
    import utils
    from utils import set_logger
    from absl import logging
    import os
    import libs.autoencoder
    import clip
    from libs.clip import FrozenCLIPEmbedder
    from libs.uvit_multi_post_ln_v1 import UViT
    from configs.unidiffuserv1 import get_config
    import builtins
    import ml_collections
    from score import Evaluator
    from torch import multiprocessing as mp
    
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda:0 default
    
    nnet = UViT(**config.nnet)
    origin_sd = torch.load("models/uvit_v1.pth", map_location='cpu')
    nnet.load_state_dict(origin_sd, strict=False)

    nnet.to(device)
    
    

    
    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device=device)
    clip_img_model, clip_img_model_preprocess = clip.load(config.clip_img_model, jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)
    
    ev = Evaluator()
    
    return {
        "device": device,
        'config': config,
        "origin_sd": origin_sd,
        "nnet": nnet,
        "autoencoder": autoencoder,
        "clip_text_model": clip_text_model,
        "clip_img_model": clip_img_model,
        "clip_img_model_preprocess": clip_img_model_preprocess,
        "ev": ev
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

    nnet = context["nnet"]
    autoencoder = context["autoencoder"]
    clip_text_model = context["clip_text_model"]
    ev = context["ev"]

    config.n_samples = 4
    config.n_iter = 5
    
    origin_images = [Image.open(i["path"]).convert('RGB') for i in json_data["source_group"]]
    origin_face_embs = [ev.get_face_embedding(i) for i in origin_images]
    origin_face_embs = [emb for emb in origin_face_embs if emb is not None]
    origin_face_embs = torch.cat(origin_face_embs)
    
    origin_clip_embs = [ev.get_img_embedding(i) for i in origin_images]
    origin_clip_embs = torch.cat(origin_clip_embs)
    
    images = []
    for caption in json_data["caption_list"]:
        config.prompt = caption
        paths = sample(config, nnet, clip_text_model, autoencoder, device, json_data["id"], output_path=image_output_path)
        # face max sim is source group self sim
        max_face_sim = (origin_face_embs @ origin_face_embs.T).mean().item()
        
        # face min sim is randon pic gened by prompt
        samples = [Image.open(sample_path).convert('RGB') for sample_path in paths]
        face_embs = [ev.get_face_embedding(sample) for sample in samples]
        face_embs  = [emb for emb in face_embs if emb is not None]
        if len(face_embs) == 0:
            print(f"no face for case{json_data['id']} caption {caption}")
            continue
        min_face_sim = (origin_face_embs @ torch.cat(face_embs).T).mean().item()
        
        # text max sim is image gened by prompt sim with prompt
        text_emb = ev.get_text_embedding(caption)
        gen_clip_embs = torch.cat([ev.get_img_embedding(i) for i in samples])
        max_text_sim = (text_emb @ gen_clip_embs.T).mean().item()
        
        
        # text min sim is source group with prompt
        min_text_sim = (text_emb @ origin_clip_embs.T).mean().item()
        
        # image reward max sim is gened by prompt sim with prompt
        max_image_reward = np.mean([ev.image_reward.score(caption, path) for path in paths ]).item()
        
        # image reward min sim is source group with prompt
        min_image_reward = np.mean([ev.image_reward.score(caption, i["path"]) for i in json_data["source_group"] ]).item()
        
        
        
        images.append({"prompt": caption, "paths": paths, 
                       "max_face_sim": max_face_sim,
                       "min_face_sim": min_face_sim,
                       "max_text_sim": max_text_sim,
                       "min_text_sim": min_text_sim,
                       "max_image_reward": max_image_reward,
                       "min_image_reward": min_image_reward,
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
    print(f"Time {name} elapsed: {TIME_ED - TIME_ST}")    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--json_data_path", type=str, default="test_json_data", help="file contains prompts")
    parser.add_argument("-j","--json_output_path", type=str, default="bound_json_outputs", help="file contains scores")
    parser.add_argument("-i","--image_output_path", type=str, default="bound_image_outputs", help="output dir for generated images")
    return parser.parse_args()

def main():
    """
    main function
    """
    arg = get_args()
    os.makedirs(arg.json_output_path, exist_ok=True)
    os.makedirs(arg.image_output_path, exist_ok=True)
    # load json files
    json_data_ls = load_json_files(arg.json_data_path)

    # process json files
    context = prepare_context()
    
    for json_data in json_data_ls:
        tik()
        out = process_one_json(json_data, arg.image_output_path, context)
        tok(f"process_one_json: {json_data['id']}")
        with open(os.path.join(arg.json_output_path, f"{json_data['id']}.json"), 'w') as f:
            json.dump(out, f, indent=4)
        
        
if __name__ == "__main__":
    main()