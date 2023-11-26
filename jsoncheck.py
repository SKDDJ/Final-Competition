import json
import argparse
import os
import logging

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--json_data_path", type=str, default="/home/schengwei/Competitionrepo/train_data/json", help="file contains prompts")
    return parser.parse_args()

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
def main():
    arg = get_args()
    # load json files
    json_data_ls = load_json_files(arg.json_data_path)
    print("json_data_ls: {}".format(len(json_data_ls)))
    
    for json_data in json_data_ls:
        logging.info(f"process json_data: {json_data['id']}")
        # print(f"process json_data: {json_data['id']}")
        image_paths = [i["path"] for i in json_data["source_group"]]
        # 获取目录中所有的图片文件
        all_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
# 按照人的编号和照片的编号对文件名进行排序
        sorted_files = sorted(all_files, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))

        print(sorted_files)
        
if __name__ == "__main__":
    main()
    print("success!")
