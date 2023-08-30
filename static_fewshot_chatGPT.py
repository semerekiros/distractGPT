import os
import json
import uuid
from chatgpt_wrapper import ChatGPT
import time
#import requests
from parser import args_parser
bot = ChatGPT()

prompts_lookup = dict()
prompts_lookup = {"en": {"answer_prefix": "answer: ", "question_prefix":"question: ", "suffix":"\nGenerate 20 incorrect answers: ", "distractor_prefix":"Incorrect answers: "},
                  "nl": {"answer_prefix": "antwoord: ", "question_prefix":"vraag: ", "suffix":"\nGenereer 20 onjuiste antwoorden: ", "distractor_prefix":"onjuiste antwoorden: "},
                  "fr":{"answer_prefix": "répondre: ", "question_prefix":"question: ", "suffix":"Générez 20 réponses incorrectes: ", "distractor_prefix":"réponses incorrectes: "}
                  }

'''
prompt = "\nGenerate 20 plausible but inherently incorrect answers: "
#prompt = "\nGenereer 20 plausibele maar onjuiste antwoorden: "
#answer_prefix = "answer"
#question_prefix = "question"
answer_prefix = "antwoord"
question_prefix = "vraag"
'''
def construct_prompt(question_text, answer, similar_questions, lang="nl"):
    print(f'choice of language is: {lang}')
    few_shots = []

    #answer_prefix, question_prefix, suffix, distractor_prefix = prompts_lookup[lang]
    prompt_details = prompts_lookup[lang]

    answer_prefix = prompt_details["answer_prefix"]
    question_prefix = prompt_details["question_prefix"]
    suffix = prompt_details["suffix"]
    distractor_prefix = prompt_details["distractor_prefix"]


    for qindex, sim_q in enumerate (similar_questions):
        question_slot = question_prefix + sim_q["question"] + "\n" + \
                        answer_prefix + sim_q["answer"] + "\n" + \
                        distractor_prefix

        for distindex, dist in enumerate(sim_q["distractors"]):
            question_slot = question_slot + "\n" + str(distindex + 1) + ". " + dist
        few_shots.append(question_slot)

    few_shot_prompt = "\n\n".join(few_shots)


    question_text = question_prefix + ": " + question_text + '\n'
    answer_text = answer_prefix + ": " + answer + '\n'
    if '###' in answer:
        answer_text = ''
        answers = answer.split("###")
        for inx, ans in enumerate(answers):
            ans = ans.strip()
            answer_text = answer_text + answer_prefix + " " + str(inx + 1) + ': ' + ans + "\n"
        # print(answer_text)

    zeroshot_prompt = question_text + answer_text + suffix

    dynamic_prompt = few_shot_prompt + "\n\n" + zeroshot_prompt


    return dynamic_prompt


def load_json(path):
    with open(path, "r") as f:
        file = json.load(f)
    return file

def get_static_questions(topk=5, lang='en'):
    pred_folder = "static-examples/"
    static_examples =''
    if lang == 'en':
        static_examples = 'static_english.json'
    else:
        static_examples = 'static_french.json'

    q_path = pred_folder + str(static_examples)

    question_details = load_json(q_path)


    similar_questions = question_details["response"]["Result"]
    return similar_questions[:topk]


def get_predicted_ids(path="predictions-few-shot/"):
    predicted_ids = []
    #path = "predictions/"
    fnames = []
    for p in os.listdir(path):
        qid = p.strip('.json')
        qid = qid.strip()
        fnames.append(qid)
    #print(fnames)

    return fnames

def dump_json(data, outpath):
    print('Saving to', outpath)
    with open(outpath, 'w') as out:
        json.dump(data, out, indent=4, separators=(',', ': '))

def get_fpaths(path="test-data/"):
    fnames = []
    for p in os.listdir(path):
        if os.path.isfile(os.path.join(path, p)):
            fnames.append(os.path.join(path, p))


    return fnames


def create_ids_dump(files):
    for fname in files:
        with open(fname, "r") as f:
            subject = json.load(f)

        for question in subject:
            new_id = uuid.uuid4()
            question["qid"] = str(new_id)
        dump_json(subject, fname)


def predict(input):
    bot.new_conversation()
    response = bot.ask(input)

    return response



def predict_from_argument(fname, lang='en'):
    skipped = 0
    with open(fname, "r") as f:
        questions = json.load(f)

    print(f'predicting for subject: {fname}')
    for question in questions:
        question_text = question["question"].strip()
        answer = question["answer"].strip()
        qid = question["qid"]
        pred_ids = get_predicted_ids("predictions-few-shot-static/")
        if qid in pred_ids:
            skipped +=1
            continue

        #question_text = "question: " + question_text + '\n'
        #answer_text = "answer: " + answer + '\n'

        similar_questions = get_static_questions(5, lang)

        input = construct_prompt(question_text, answer, similar_questions, lang=lang)


        #time.sleep(5)
        result = predict(input)
        if "Unusable response produced" in result:
            print(f'Sth has gone wrong. Error message {result}')
            break
        #result = "hi"
        one_prediction = {}
        one_prediction['qid'] = qid
        one_prediction['input'] = input
        one_prediction['response'] = result
        out_path = "predictions-few-shot-static/" + qid + ".json"

        dump_json(one_prediction, out_path)

    print(f'total skipped {skipped}')




if __name__ == '__main__':
    args = args_parser()
    args = args.parse_args()
    fname = args.subject
    lang = args.lang
    predict_from_argument(fname, lang)
    #pred_ids = get_predicted_ids()
    #with open("test-data/biology.json") as f:
     #   bio = json.load(f)
    #first_id = bio[0]['qid']
    #if first_id in pred_ids:
     #   print(f'found id {first_id}')