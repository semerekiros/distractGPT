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


    parser.add_argument("--subject",
                        dest="subject",
                        type=str,
                        default="./test-data/biology.json",
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

    subject_path = args.subject

    subject_json = read_json(subject_path)

    uuid_mapping = {}
    for inx, q_info in enumerate(subject_json):
        uuid_mapping [inx] = q_info["qid"]



    destination_folder= "processed/mapping/"

    fname = subject_path.split("/")[-1]
    fname = fname.split(".json")[0]


    dest_path = destination_folder + fname + "_mapping.json"

    #dump_json(uuid_mapping, dest_path)

    all_mapping = {}
    parent_folder = "processed/mapping/"
    all_subjects = get_all_fnames(parent_folder)

    for subname in all_subjects:
        path = parent_folder + subname
        sub_json = read_json(path)

        key_name = subname.split("_mapping.json")[0]
        all_mapping[key_name] = sub_json
    final_file = parent_folder + "global_mapping.json"
    dump_json(all_mapping, final_file)



