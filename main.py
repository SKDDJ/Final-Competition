
import json
import os
import time
import signal
import argparse
# from torch import multiprocessing


TIME_ST = time.time()
TIME_ED = time.time()
TOTAL_TIME = 0


from load_model import prepare_context, process_one_json, accelerate_model

def handler(signum, frame):
    raise Exception("end of time")


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



def tik():
    global TIME_ST
    TIME_ST = time.time()
def tok(name):
    global TIME_ED
    TIME_ED = time.time()
    print(f"Time {name} elapsed: {TIME_ED - TIME_ST}")    
def tik_tok():
    global TOTAL_TIME
    TOTAL_TIME = TOTAL_TIME + TIME_ED - TIME_ST

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--json_data_path", type=str, default="final_json_data/json", help="file contains prompts")
    parser.add_argument("-j","--json_output_path", type=str, default="json_outputs", help="file contains scores")
    parser.add_argument("-i","--image_output_path", type=str, default="image_outputs", help="output dir for generated images")
    return parser.parse_args()

def main():

    arg = get_args()
    os.makedirs(arg.json_output_path, exist_ok=True)
    os.makedirs(arg.image_output_path, exist_ok=True)
    # load json files
    json_data_ls = load_json_files(arg.json_data_path)

    # process json files
    context = prepare_context()
    

    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60*90)
    
    try:
        for json_data in json_data_ls:
            tik()
            out = process_one_json(json_data, arg.image_output_path, context)
            tok(f"process_one_json: {json_data['id']}")
            tik_tok()
            with open(os.path.join(arg.json_output_path, f"{json_data['id']}.json"), 'w') as f:
                json.dump(out, f)
    except Exception as e:
        print(e)

    # # Wait for the accelerate_model process to finish
    # process.join()
    print(f"\033[91m Total Time elapsed: {TOTAL_TIME}\033[00m")
if __name__ == "__main__":
    

    main()



