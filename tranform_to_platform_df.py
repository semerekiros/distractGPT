import csv
import json
import numpy as np
import os
import uuid
from tqdm import tqdm
import time
from parser import args_parser


import csv
import json
import numpy as np
import os
import uuid
from tqdm import tqdm
import time
from parser import args_parser

def write_to_file(outpath, data):
    print('Saving to', outpath)
    with open(outpath, 'w') as out:
        json.dump(data, out, indent=4, separators=(',', ': '))

class PlatformDataType():
    def __init__(self, data_root, subname, modelname, topk):

        self.topk = topk
        self.final_data = dict()
        self.jsonf = dict()
        self.metadata = dict()

        self.content = list()

        with open(data_root, "r") as f:
            self.json_data = json.load(f)

        self.create_metadata(subname)  # create the meta data information

        self.add_predictions(modelname)

        self.final_data["meta_data"] = self.metadata
        self.final_data["content"] = self.content

    def create_metadata(self, subname):
        self.metadata["session_id"] = "all"
        self.metadata["subject"] = subname
        self.metadata["level"] = "all"
        self.metadata["estimated_duration"] = "unknown"
        self.metadata["task_type"] = "all"

    def transform_gtruth(self):
        self.content = list()

        for i, inst in enumerate(self.json_data):

            one_mcq = dict()
            ground_truth = list()

            answers = inst['answer'].strip("\n")
            question = inst['question'].strip("\n")
            distractors = inst['distractors']
            # language = inst["language"]
            uuid_qid = inst["qid"]

            one_mcq["qid"] = uuid_qid
            one_mcq["question"] = question
            one_mcq["answers"] = answers

            # one_mcq["language"] = language
            dist_ctr = 0

            for dist in distractors:
                one_distractor = dict()
                dist = dist.strip("\n")
                dist = dist.strip()

                if len(dist) != 0:
                    dist_ctr = dist_ctr + 1

                    one_distractor["distid"] = dist_ctr
                    one_distractor["distractor"] = dist

                    one_distractor["modelid"] = "human"
                    one_distractor["score"] = 1

                    ground_truth.append(one_distractor)

            one_mcq["ground_truth"] = ground_truth
            self.content.append(one_mcq)

    def add_predictions(self, modname):
        self.transform_gtruth()  # transform the ground truth to the ground truth format of the final data type of the annotation platfrom

        for index, q_info in enumerate(self.content):
            predictions = list()

            question = q_info["question"]
            answer = q_info['answers']
            qid = q_info["qid"]

            # self.model_predictions, prompt = self.get_predictions(qid, modname, topk=self.topk)
            self.model_predictions = self.get_predictions(qid, modname, topk=self.topk)

            # output = self.model_predictions

            # print(f'Len (model.predictions), {len(self.model_predictions)}')
            # print("__________________________________________________________________ \n")
            # print("question: ", question)
            # print("answer: " , answer)
            # print("prompt: ", prompt)

            for dist, score in self.model_predictions:
                # print(dist, score)

                # for dist, score in output:

                #print(index, dist, score)

                dist_info = dict()

                d_id = uuid.uuid4()
                dist_info["distid"] = str(d_id)
                dist_info["distractor"] = dist
                dist_info["modelid"] = modname
                dist_info["score"] = score
                predictions.append(dist_info)
            # print("__________________________ENNNNNNNNNNNNNNNNNNDDDDDDDDDD___________________________ \n")
            self.content[index]["proposed_distractors"] = predictions

    def get_predictions(self, qid, modname, topk):

        assert modname in ["few-shot", "zero-shot", "mt5", "few-shot-static"], f"modname expects few-shot, zero-shot or mt5 model names but, got: {modname}"

        predictions_folder = "predictions-" + str(modname)
        json_fname = predictions_folder + "/" + str(qid) + ".json"
        with open(json_fname, "r") as f:
            result = json.load(f)

        predicted_distractors = result["response"]
        predicted_distractors = predicted_distractors.split("\n")
        cleaned_distractors = []

        for i in predicted_distractors:
            d = i.strip()
            d = d.strip("\n")
            if len(d) != 0:
                d = d.split(" ")
                rank = d[0]
                body = " ".join(d[1:])
                # print(f'rank: {rank} body: {body}')

                cleaned_distractors.append((body, rank))

        # return cleaned_distractors[:topk],  result["input"]
        return cleaned_distractors[:topk]

    def get_json(self):

        return self.final_data

    def total_question(self):
        return len(self.content)

    @staticmethod
    def get_all_fnames(folder):
        files = list()
        fpaths = list()
        fpaths.append(folder)

        for fpath in tqdm(fpaths, desc="Looping over fpaths"):
            files.extend(os.listdir(fpath))
        return files


if __name__ == "__main__":
    args = args_parser()
    args = args.parse_args()
    fname = args.subject
    processed_path = "processed/"



    #model = "mt5"
    #model = "zero-shot"
    #model = "few-shot"
    model = "few-shot-static"

    subject_parent = "test-data/"
    subject = "french.json"

    subject_path = subject_parent + subject

    platform_data = PlatformDataType(subject_path, subject, model, topk=10)
    final_data = platform_data.get_json()
    print(final_data["meta_data"])

    parent_dir = processed_path + model

    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    target_path = parent_dir + "/" + subject

    write_to_file(target_path, final_data)

