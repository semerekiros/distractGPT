import json
import random
import os
import argparse
from tqdm import tqdm
def args_parser():
    parser = argparse.ArgumentParser(description="Combine predictions from different models to be displayed in the annotation platfrom")


    parser.add_argument("--subject",
                        dest="subject",
                        type=str,
                        default="./processed/",
                        help="subject")

    return parser




def dump_json(data, outpath):
    print('Saving to', outpath)
    with open(outpath, 'w') as out:
        json.dump(data, out, indent=4, separators=(',', ': '))
def read_json(target_file):
    with open(target_file, "r") as fp:
        loaded_json = json.load(fp)

    return loaded_json


#job = {jid: info}
#info = dict ()   dict_info = dict() dict_info['subject'] ='naturalsciences12' dict_info['task'] = 'comparison' dict_info['qids'] = qids new_jobs['4'] = dict_info
#class Jobs():
 #  def __init__(self, data_root, subname, modelname, topk):

def get_all_fnames(folder):
    files = list()
    fpaths = list()
    fpaths.append(folder)

    for fpath in tqdm(fpaths, desc="Looping over fpaths"):
        files.extend(os.listdir(fpath))
    return files
def create_job(qids, subject):
    dict_info = dict()
    dict_info['subject'] =subject
    dict_info['task'] = 'comparison'
    dict_info['qids'] = qids

    return dict_info

def create_admin_job():
    job = {}
    parent_path = "./test-data"
    mapping_qids = read_json("processed/mapping/global_mapping.json")
    all_subjects = get_all_fnames(parent_path)

    job_id = 100

    for sname in all_subjects:
        file_path = os.path.join(parent_path, sname)
        subject_json = read_json(file_path)

        subject_qids = []
        for ix, inst in enumerate(subject_json):
            old_qid = inst["qid"]
            subject_key = sname.split(".json")[0]
            subject_mapping = mapping_qids.get(subject_key)
            for k, v in subject_mapping.items():
                if old_qid == v:
                    new_id = k
            subject_qids.append(new_id)

        random.shuffle(subject_qids)

        left = subject_qids[:25]
        right = subject_qids[25:]
        sname_stripped = sname.split(".json")
        sname_stripped = sname_stripped[0]
        formatted_qids = create_job(subject_qids, sname_stripped)
        indexer = str(job_id)
        job[indexer] = formatted_qids
        job_id = job_id + 100

    dump_json(job, "processed/jobs.json")

def create_jobs_teachers(jobs_file, limits, sname_path):

    job_id = 100

    #file_path = os.path.join(parent_path, sname)
    subject_json = read_json(sname_path)
    mapping_qids = read_json("processed/mapping/global_mapping.json")

    subject_qids = []
    for ix, inst in enumerate(subject_json):
        old_qid = inst["qid"]
        subject_key = sname_path.split("/")[-1].split(".json")[0]
        subject_mapping = mapping_qids.get(subject_key)
        for k, v in subject_mapping.items():
            if old_qid == v:
                new_id = k
        subject_qids.append(new_id)

    random.shuffle(subject_qids)

    left = subject_qids[:25]
    right = subject_qids[25:]

    sname_stripped = sname_path.split("/")[-1].split(".json")[0]
    #sname_stripped = sname_stripped.split("/")[-1]

    formatted_qids_left = create_job(left, sname_stripped)
    formatted_qids_right = create_job(right, sname_stripped)

    start = limits[0]
    end =limits[1]
    for i in range(start, end+1):
        indexer = str(i)
        if (i % 2) == 0:
            jobs_file[indexer] = formatted_qids_left
        else:
            jobs_file[indexer] = formatted_qids_right

    dump_json(jobs_file, "processed/jobs.json")

if __name__ == '__main__':
    random.seed(42)
    '''
    uncomment create_admin_job() and comment everything else to create initial job.json and add new jobs by reversing this one!

    
    '''

    #create_admin_job()

    #'''
    jobs_list = {"english":[1,10],
                 "french": [11,20],
                 "naturalsciences": [21,30],
                 "history": [31,40],
                 "biology": [41,50],
                 "geography": [51,60]}



    args = args_parser()
    args = args.parse_args()

    existing_jobs = read_json("processed/jobs.json")

    subjectfpath = args.subject
    print(f'subjectpath: {subjectfpath}')
    subject = subjectfpath.split("/")[-1]
    subject = subject.split(".json")[0]
    print(f'subjectpath after stripping json: {subject}')

    limits = jobs_list[subject]
    start = limits[0]
    end = limits[1]

    
    create_jobs_teachers(existing_jobs, limits, subjectfpath)
    #'''











