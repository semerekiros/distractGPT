import os
import json
import uuid
from chatgpt_wrapper import ChatGPT
import time
from parser import args_parser
bot = ChatGPT()

prompt = "\nGenerate 20 plausible but inherently incorrect answers: "
answer_prefix = "answer"
question_prefix = "question"

#prompt = "\nGenereer 20 plausibele maar onjuiste antwoorden: "
#answer_prefix = "antwoord"
#question_prefix = "vraag"

#prompt = "\nGénérez 20 réponses plausibles mais intrinsèquement incorrectes : "
#answer_prefix = "répondre"
#question_prefix = "question"

def get_predicted_ids():
    predicted_ids = []
    path = "predictions/"
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


#create_ids_dump(files)
def predict_all_folder():
    files = get_fpaths("test-data/")
    for fname in files:
        with open(fname, "r") as f:
            questions = json.load(f)
        for question in questions:
            question_text = question["question"].strip()
            answer = question["answer"].strip()
            qid= question["qid"]

            question_text = question_prefix + ": " + question_text + '\n'
            answer_text = answer_prefix + ": " + answer + '\n'
            if '###' in answer:
                answer_text = ''
                answers = answer.split("###")
                for inx, ans in enumerate(answers):
                    ans = ans.strip()
                    answer_text = answer_text + answer_prefix+ ' ' + str(inx+1) + ': ' + ans + "\n"
                #print(answer_text)
            input = question_text + answer_text + prompt

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
            out_path = "predictions/" + qid + ".json"

            dump_json(one_prediction, out_path)


def predict_from_argument(fname):
    skipped = 0
    with open(fname, "r") as f:
        questions = json.load(f)

    print(f'predicting for subject: {fname}')
    for question in questions:
        question_text = question["question"].strip()
        answer = question["answer"].strip()
        qid = question["qid"]
        pred_ids = get_predicted_ids()
        if qid in pred_ids:
            skipped +=1
            continue

        #question_text = "question: " + question_text + '\n'
        #answer_text = "answer: " + answer + '\n'

        question_text = question_prefix + ": " + question_text + '\n'
        answer_text = answer_prefix + ": " + answer + '\n'
        if '###' in answer:
            answer_text = ''
            answers = answer.split("###")
            for inx, ans in enumerate(answers):
                ans = ans.strip()
                answer_text = answer_text + answer_prefix + " " + str(inx + 1) + ': ' + ans + "\n"
            # print(answer_text)
        input = question_text + answer_text + prompt

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
        out_path = "predictions/" + qid + ".json"

        dump_json(one_prediction, out_path)

    print(f'total skipped {skipped}')




if __name__ == '__main__':
    args = args_parser()
    args = args.parse_args()
    fname = args.subject
    predict_from_argument(fname)
    #pred_ids = get_predicted_ids()
    #with open("test-data/biology.json") as f:
     #   bio = json.load(f)
    #first_id = bio[0]['qid']
    #if first_id in pred_ids:
     #   print(f'found id {first_id}')