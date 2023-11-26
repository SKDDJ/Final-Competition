import os
import json
import glob
import numpy as np
import torch
import clip
from PIL import Image
import argparse
import warnings
from score_utils.face_model import FaceAnalysis
import ImageReward as RM
from typing import Union
from configs.unidiffuserv1 import get_config

warnings.filterwarnings("ignore")

class Evaluator():
    def __init__(self):
        config = get_config()
        config.device = "cuda:1"
        self.clip_device = config.device
        # self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.clip_device
        self.clip_model, self.clip_preprocess = clip.load("other_models/clip/ViT-B-32.pt", device=self.clip_device)
        self.clip_tokenizer = clip.tokenize
        
        self.face_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # self.image_reward = RM.load("ImageReward-v1.0")

        self.image_reward = RM.load("./ImageReward/ImageReward.pt", med_config="./ImageReward/med_config.json")
        self.face_model.prepare(ctx_id=0, det_size=(512, 512))
        

    def pil_to_cv2(self, pil_img):
        return np.array(pil_img)[:,:,::-1]
    
    def get_face_embedding(self, img):
        """ get face embedding
        """
        if type(img) is not np.ndarray:
            img = self.pil_to_cv2(img)
            
        faces = self.face_model.get(img, max_num=1) ## only get first face
        if len(faces) <= 0:
            return None
        else:
            emb = torch.Tensor(faces[0]['embedding']).unsqueeze(0)
            emb /= emb.norm(dim=-1, keepdim=True)
            return emb
            

    def sim_face(self, img1, img2):
        """ 
        calcualte face similarity using insightface
        """
        if type(img1) is not np.ndarray:
            img1 = self.pil_to_cv2(img1)
        if type(img2) is not np.ndarray:
            img2 = self.pil_to_cv2(img2)
            
        feat1 = self.get_face_embedding(img1)
        feat2 = self.get_face_embedding(img2)
        
        if feat1 is None or feat2 is None:
            return 0
        else:
            similarity = feat1 @ feat2.T
            return similarity.item()
        
    def sim_face_emb(self, img1, embs):
        """ 
        calcualte face similarity using insightface
        """
        if type(img1) is not np.ndarray:
            img1 = self.pil_to_cv2(img1)
            
        feat1 = self.get_face_embedding(img1)
        
        if feat1 is None:
            return 0
        else:
            similarity = feat1 @ embs.T
            return similarity.mean().item()
    
    def get_img_embedding(self, img):
        """ 
        get clip image embedding
        """
        x = self.clip_preprocess(img).unsqueeze(0).to(self.clip_device)
        with torch.no_grad():
            feat = self.clip_model.encode_image(x)
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat

    def get_text_embedding(self, text):
        """ 
        get clip image embedding
        """
        x = self.clip_tokenizer([text]).to(self.clip_device)
        with torch.no_grad():
            feat = self.clip_model.encode_text(x)
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat
        
 
        
    def sim_clip_text(self, img, text):
        """ 
        calcualte img text similarity using CLIP
        """
        feat1 = self.get_img_embedding(img)
        feat2 = self.get_text_embedding(text)
        similarity = feat1 @ feat2.T
        return max(0,similarity.item())
    
    

    
def read_img_pil(p):
    return Image.open(p).convert("RGB")



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

def pre_check(source_json_dir, gen_json_dir, bound_json_dir):
    """
    1. check common ids
    2. check enough images
    3. return list of tuple (source_json, gen_json, bound_json)
    """
    
    id_to_source_json = {json_data["id"]: json_data for json_data in load_json_files(source_json_dir)}
    id_to_gen_json = {json_data["id"]: json_data for json_data in load_json_files(gen_json_dir)}
    id_to_bound_json = {json_data["id"]: json_data for json_data in load_json_files(bound_json_dir)}
    
    common_ids = set(id_to_source_json.keys()) & set(id_to_gen_json.keys())
    
    print(f"共有{len(common_ids)}个id")
    
    case_pair_ls = []
    for id in common_ids:
        source_json = id_to_source_json[id]
        gen_json = id_to_gen_json[id]
        bound_json = id_to_bound_json[id]
        
        
        for idx, item in enumerate(gen_json["images"]):
            if item["prompt"] not in source_json["caption_list"]:
                print(f"prompt {item['prompt']} not in source json")
                gen_json["images"].remove(item)
            if len(item["paths"]) != 4:
                print(f"delete item {item}")
                gen_json["images"].remove(item)
        case_pair_ls.append((source_json, gen_json, bound_json))
    return case_pair_ls

def score(ev, source_json, gen_json, bound_json, out_json_dir):
    
    
    # get ref images
    ref_image_paths = [ i["path"] for i in source_json["source_group"]]
    ref_face_embs = [ev.get_face_embedding(read_img_pil(i)) for i in ref_image_paths]
    ref_face_embs  = [emb for emb in ref_face_embs if emb is not None] # remove None
    ref_face_embs = torch.cat(ref_face_embs)

    text_ac_scores = 0
    face_ac_scores = 0
    image_reward_ac_scores = 0
    image_reward_ac_decrease = 0
    
    normed_text_ac_scores = 0
    normed_face_ac_scores = 0
    normed_image_reward_ac_scores = 0
    normed_image_reward_ac_decrease = 0
    out_json = {"id": gen_json["id"], "images": []}
    commom_prompts = set([item["prompt"] for item in gen_json["images"]]) & set([item["prompt"] for item in bound_json["images"]])
    prompt_to_item = {item["prompt"]: item for item in gen_json["images"]}
    bound_prompt_to_item = {item["prompt"]: item for item in bound_json["images"]}
    if len(commom_prompts) != len(bound_json["images"]):
        print(f"共有{len(commom_prompts)}个prompt, bound json有{len(bound_json['images'])}个prompt")
        print(bound_json)
    
    for prompt in commom_prompts:
        item = prompt_to_item[prompt]
        bound_item = bound_prompt_to_item[prompt]
        
        assert item["prompt"] == bound_item["prompt"], f"prompt {item['prompt']} not equal to bound prompt {bound_item['prompt']}"
        if len(item["paths"]) < 4:
            continue
        
        # clip text similarity
        samples = [read_img_pil(sample_path) for sample_path in item["paths"]]
        scores_text = [ev.sim_clip_text(sample, item["prompt"]) for sample in samples]
        mean_text = np.mean(scores_text)
        
        # image reward
        scores_image_reward = [ev.image_reward.score(item["prompt"], sample_path) for sample_path in item["paths"]]
        mean_image_reward = np.mean(scores_image_reward)
        
        # hps v2
        # scores_hpsv2 = [ev.hpsv2_score(sample, item["prompt"])[0].item() for sample in samples]
        # mean_hpsv2 = np.mean(scores_hpsv2)
        
        # face similarity
        sample_faces = [ev.get_face_embedding(sample) for sample in samples]
        sample_faces = [emb for emb in sample_faces if emb is not None] # remove None
        if len(sample_faces) <= 1:
            print("too few faces")
            continue
        scores_face = [(sample_face @ ref_face_embs.T).mean().item() for sample_face in sample_faces]
        mean_face = np.mean(scores_face)
        
        subed_score_text = mean_text - bound_item["min_text_sim"]
        subed_score_face = mean_face - bound_item["min_face_sim"]
        subed_image_reward = mean_image_reward - bound_item["min_image_reward"]
        image_reward_decrease = bound_item["max_image_reward"] - mean_image_reward
        
        
        normed_score_text = subed_score_text / (bound_item["max_text_sim"] - bound_item["min_text_sim"])
        normed_score_face = subed_score_face / (bound_item["max_face_sim"] - bound_item["min_face_sim"])
        normed_score_image_reward = subed_image_reward / (bound_item["max_image_reward"] - bound_item["min_image_reward"])
        normed_image_reward_decrease = image_reward_decrease / (bound_item["max_image_reward"] - bound_item["min_image_reward"])
        
        if normed_score_image_reward < 0.1:
            # print(f"Image reward too low for prompt: '{item['prompt']}' in item: {item}")
            print(f"\033[91mface similarity too low for prompt:\033[0m '{item['prompt']}' in item(id):\033[91m{gen_json['id']}\033[0m")
            # print("too low image reward")
            continue
        if normed_score_face < 0.1:
            print(f"\033[91mface similarity too low for prompt:\033[0m '{item['prompt']}' in item(id):\033[91m{gen_json['id']}\033[0m")
            # print(f"face similarity too low for prompt: '{item['prompt']}' in item: {item}")
            # print("too low face similarity")
            continue
        
        normed_text_ac_scores += normed_score_text
        normed_face_ac_scores += normed_score_face
        normed_image_reward_ac_scores += normed_score_image_reward
        normed_image_reward_ac_decrease += normed_image_reward_decrease
        
        face_ac_scores += subed_score_face
        text_ac_scores += subed_score_text
        image_reward_ac_scores += subed_image_reward
        image_reward_ac_decrease += image_reward_decrease
        
        out_json["images"].append({"prompt": item["prompt"], 
                                   "scores_text": scores_text, 
                                   "scores_face": scores_face, 
                                   "scores_image_reward": scores_image_reward,
                                #    "scores_hpsv2": scores_hpsv2,
                                   "subed_score_text": subed_score_text,
                                   "subed_score_face": subed_score_face,
                                    "subded_image_reward": subed_image_reward,
                                    "image_reward_decrease": image_reward_decrease,
                                    
                                    "normed_score_text": normed_score_text,
                                    "normed_score_face": normed_score_face,
                                    "normed_score_image_reward": normed_score_image_reward,
                                    "normed_image_reward_decrease": normed_image_reward_decrease})
        
    with open(os.path.join(out_json_dir, f"{gen_json['id']}.json"), 'w') as f:
        json.dump(out_json, f, indent=4)
        
    return {"text_ac_scores":text_ac_scores, 
            "face_ac_scores":face_ac_scores,
            "image_reward_ac_scores":image_reward_ac_scores,
            "image_reward_ac_decrease":image_reward_ac_decrease,
            
            "normed_text_ac_scores":normed_text_ac_scores,
            "normed_face_ac_scores":normed_face_ac_scores,
            "normed_image_reward_ac_scores":normed_image_reward_ac_scores,
            "normed_image_reward_ac_decrease":normed_image_reward_ac_decrease,
            }
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--source_json_dir', type=str, default='final_json_data/json', help='task json files, original json data to generate images.')
    
    
    # parser.add_argument('--gen_json_dir', type=str, default='aaaaaaasaveresults/aaaajsons', help='json after generating images.')
    # parser.add_argument("--out_json_dir", type=str, default="aaaaaaasaveresults/aaaascores", help="score json ouput")
        
    parser.add_argument('--gen_json_dir', type=str, default='aaaaaaasaveresults/bbbbjsons', help='json after generating images.')
    parser.add_argument("--out_json_dir", type=str, default="aaaaaaasaveresults/bbbbscores", help="score json ouput")
    
    
    parser.add_argument("--bound_json_dir", type=str, default="abase_json_outputs", help="baseline score json ouput")

    args = parser.parse_args()
    os.makedirs(args.out_json_dir, exist_ok=True)
    
    pairs = pre_check(args.source_json_dir, args.gen_json_dir, args.bound_json_dir)
    
    ev = Evaluator()
    total_text_score = 0
    total_face_score = 0
    total_image_reward_score = 0
    total_image_reward_decrease = 0
    normed_total_text_score = 0
    normed_total_face_score = 0
    normed_total_image_reward_score = 0
    normed_total_image_reward_decrease = 0
    for source_json, gen_json, bound_json in pairs:
        rt_dict = score(ev, source_json, gen_json, bound_json, args.out_json_dir)
        
        total_text_score += rt_dict["text_ac_scores"]
        total_face_score += rt_dict["face_ac_scores"]
        total_image_reward_score += rt_dict["image_reward_ac_scores"]
        total_image_reward_decrease += rt_dict["image_reward_ac_decrease"]
        
        normed_total_text_score += rt_dict["normed_text_ac_scores"]
        normed_total_face_score += rt_dict["normed_face_ac_scores"]
        normed_total_image_reward_score += rt_dict["normed_image_reward_ac_scores"]
        normed_total_image_reward_decrease += rt_dict["normed_image_reward_ac_decrease"]
    
    print(f"""
total_text_score:           {total_text_score:.4f},
total_face_score:           {total_face_score:.4f},
total_image_reward_score:   {total_image_reward_score:.4f},
total_image_reward_decrease:{total_image_reward_decrease:.4f},

normed_total_text_score:           {normed_total_text_score:.4f},
normed_total_face_score:           {normed_total_face_score:.4f},
normed_total_image_reward_score:   {normed_total_image_reward_score:.4f},
normed_total_image_reward_decrease:{normed_total_image_reward_decrease:.4f},
          """)
    