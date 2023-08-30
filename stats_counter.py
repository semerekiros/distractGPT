import json
import random
import os
import argparse
from tqdm import tqdm
def get_all_fnames(folder):
    files = list()
    fpaths = list()
    fpaths.append(folder)

    for fpath in tqdm(fpaths, desc="Looping over fpaths"):
        files.extend(os.listdir(fpath))
    return files
def args_parser():
    parser = argparse.ArgumentParser(description="Combine predictions from different models to be displayed in the annotation platfrom")


    parser.add_argument("--input",
                        dest="input",
                        type=str,
                        default="./test-data/biology.json",
                        help="path to the subject")
    parser.add_argument("--out",
                        dest="out",
                        type=str,
                        default="./test-data/",
                        help="path to the subject")

    return parser




def dump_json(data, outpath):
    print('Saving to', outpath)
    with open(outpath, 'w') as out:
        json.dump(data, out, indent=4, separators=(',', ': '))

def read_json(target_file):
    with open(target_file, "r") as fp:
        loaded_json = json.load(fp)

    return loaded_json




if __name__ == '__main__':

    args = args_parser()
    args = args.parse_args()

    subject_path = args.input
    #out_path = args.out

    subject_json = read_json(subject_path)


    fname = subject_path.split("/")[-1]
    fname = fname.split(".json")[0]

    #all_mapping = read_json("processed/mapping/global_mapping.json")

    #subject_mapping = all_mapping.get(fname)

    #new_json = subject_json.copy()
    proposed_count = 0
    for con in subject_json["content"]:
        proposed_distractors = con["proposed_distractors"]
        proposed_count = proposed_count + len(proposed_distractors)

    print(f'proposed_count: {proposed_count}')


