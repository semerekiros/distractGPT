import os
import json
import uuid
#from chatgpt_wrapper import ChatGPT
import time
import requests
from parser import args_parser
from tqdm import tqdm
#bot = ChatGPT()

prompt = "\nGenerate 20 plausible but inherently incorrect answers: "
#prompt = "\nGenereer 20 plausibele maar onjuiste antwoorden: "
#answer_prefix = "answer"
#question_prefix = "question"
answer_prefix = "antwoord"
question_prefix = "vraag"


def get_prediction_question_qsim(question, answer, topk=5, lang_filter=False):
    #print("Qsim model being used")
    headers = {
        'Content-Type': 'application/json',
    }
    api_url = 'http://193.190.127.236:5083/process'  # model1_all distractors

    data = dict()
    data["query"] = question
    data['answer'] = answer
    data["topk"] = topk

    response = requests.post(api_url, headers=headers, json=data)
    result = response.json()

    return result



def dump_json(data, outpath):
    #print('Saving to', outpath)
    with open(outpath, 'w') as out:
        json.dump(data, out, indent=4, separators=(',', ': '))

def get_fpaths(path="test-data/"):
    fnames = []
    for p in os.listdir(path):
        if os.path.isfile(os.path.join(path, p)):
            fnames.append(os.path.join(path, p))


    return fnames



def predict_all_folder():
    files = get_fpaths("test-data/")
    for fname in tqdm(files, desc="files", position=0):
        with open(fname, "r") as f:
            questions = json.load(f)
        for question in tqdm(questions, desc="questions", position=1, leave=False):
            question_text = question["question"].strip()
            answer = question["answer"].strip()
            qid= question["qid"]
            '''
            question_text = question_prefix + ": " + question_text + '\n'
            answer_text = answer_prefix + ": " + answer + '\n'
           
            if '###' in answer:
                answer_text = ''
                answers = answer.split("###")
                for inx, ans in enumerate(answers):
                    ans = ans.strip()
                    answer_text = answer_text + answer_prefix+ ' ' + str(inx+1) + ': ' + ans + "\n"
            
            input = question_text + answer_text + prompt
            '''

            result = get_prediction_question_qsim(question_text, answer, topk=10)
            if not result:
                print(f'Sth has gone wrong. Error message empty {result}')
                break
            #result = "hi"
            one_prediction = {}
            one_prediction['qid'] = qid
            one_prediction['response'] = result
            out_path = "predictions-qsim/" + qid + ".json"

            dump_json(one_prediction, out_path)



if __name__ == '__main__':
    args = args_parser()
    args = args.parse_args()
    fname = args.subject
    #predict_from_argument(fname)
    predict_all_folder()
    #pred_ids = get_predicted_ids()
    #with open("test-data/biology.json") as f:
     #   bio = json.load(f)
    #first_id = bio[0]['qid']
    #if first_id in pred_ids:
     #   print(f'found id {first_id}')