import os
import json
import uuid
import time
from parser import args_parser
from transformers import AutoTokenizer, MT5Model, MT5ForConditionalGeneration
import re
from torch import cuda
import torch
import random
prompt = "\nGenerate 20 plausible but inherently incorrect answers: "
answer_prefix = "answer"
question_prefix = "question"



class Mt5Finetuned:
    def __init__(self, tokenizer_checkpoint= 'google/mt5-base', model_path= 'checkpoints/template_based_updated/'):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path)
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        
            
        self.templates = {"en": "Which one of the following are incorrect answers?",
                     "nl": "Welke van de volgende zijn onjuiste antwoorden?",
                     "fr": "Parmi les réponses suivantes, lesquelles sont incorrectes ?"}
        self.masking_prefix = "<extra_id_"
        
    def build_input(self, question, answer, lang = "nl", no_distractors=10, version1=True):
        
        template = self.templates[lang]
        constructed_input = ""
        distractor_part = ""

        if version1:
            question_answer =  question + " </s> " + "\n1. "+ answer
        else:
            question_answer =  question + " </s> " + template + " </s> "  + "\n1. "+ answer

       
        for mask_idx in range(0, no_distractors):
            
            if version1:
                distractor_part = "\n" + distractor_part + str(mask_idx+2) + ". " + self.masking_prefix + str(mask_idx) + ">" 
            else:
                distractor_part =  distractor_part + "\n" + str(mask_idx+2) + ". " + self.masking_prefix + str(mask_idx) + ">" 


        constructed_input = question_answer + distractor_part

        return constructed_input
    
    @staticmethod
    def split_on_digit_dot_space(input_str):
        # Split the input string on the pattern of a digit, a dot, and a space

        split_list = re.split(r'\d+\. ', input_str)

        #split_list = re.split('[0-9]\. ', input_str)

        # Remove any empty strings from the list
        split_list = [text.strip() for text in split_list if text.strip()]
        return split_list
    
    def clean_text(self, inp_text):
        distractors_list = self.split_on_digit_dot_space(inp_text)
        remove_token = "<extra_"
        
        distractors_list = [dist for dist in distractors_list if not dist.startswith(remove_token)]
        
        cleaned_inp = "\n"
        #"\n\n1. La souris\n2.
        #'\n\n1. hi\n2. there\n3. how'

        for ix, dist in enumerate(distractors_list):
            cleaned_inp = cleaned_inp + "\n" + str(ix+1) + ". " + dist   
        
        return cleaned_inp
    def tensorize(self, test_sent):
        test_tokenized = self.tokenizer.batch_encode_plus([test_sent], max_length= 512, pad_to_max_length=True, return_tensors='pt')
        test_input_ids  = test_tokenized["input_ids"]
        test_attention_mask = test_tokenized["attention_mask"]
        
        test_input_ids = test_input_ids.to(self.device, dtype=torch.long)
        test_attention_mask = test_attention_mask.to(self.device, dtype=torch.long)

    
        return test_input_ids, test_attention_mask
    
    def greedy_decode(self, test_sent):
        
        
        test_input_ids, test_attention_mask = self.tensorize(test_sent)
        
        self.model.eval()

        greedy_output = self.model.generate(
            input_ids = test_input_ids,
            attention_mask = test_attention_mask,
            max_length = 150,
            num_beams=50,
            repetition_penalty = 2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        result =  self.tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        predicted_summary = self.clean_text(result)

        return result
    

    def beam_decode(self, test_sent, topk=10):
        test_input_ids, test_attention_mask = self.tensorize(test_sent)
        self.model.eval()

        beam_outputs = self.model.generate(
            input_ids = test_input_ids,
            attention_mask = test_attention_mask,
            max_length = 150,
            num_beams=20,
            num_return_sequences=topk,
            repetition_penalty = 2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        predictions = []

        for beam_output in beam_outputs:
            predicted_summary = self.tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predicted_summary = self.clean_text(predicted_summary)
            predictions.append(predicted_summary)

        return predictions

    def nucleus_decode(self, test_sent, topk=10):
        test_input_ids, test_attention_mask = self.tensorize(test_sent)
        self.model.eval()

        beam_outputs = self.model.generate(
            input_ids = test_input_ids,
            attention_mask = test_attention_mask,            
            do_sample=True,                 
            max_length=30,                    
            top_p=0.84,                      
            top_k=80,                             
            num_return_sequences=10,                                      
            min_length=3,
            temperature=0.9,
            repetition_penalty=2.5,   
            length_penalty=1.5,    
            no_repeat_ngram_size=2,

        )

        predictions = []
        for beam_output in beam_outputs:
            predicted_summary = self.tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predicted_summary = self.clean_text(predicted_summary)
            predictions.append(predicted_summary)
        return predictions
    
    def round_robin_selection(self, test_sent, topk=10):
        
        selected_distractors = []
        
        nucleus_preds = self.nucleus_decode(test_sent, topk)
        greedy_preds = self.beam_decode(test_sent, topk)
        
        zipped_preds = zip(greedy_preds, nucleus_preds)
        
        for greed, nuc in zipped_preds:
            #greed_list = greed.strip().split("\n")
            #nuc_list   = nuc.strip().split("\n")
            greed_list = self.split_on_digit_dot_space(greed.strip())
            nuc_list   = self.split_on_digit_dot_space(nuc.strip())
            
            if len(selected_distractors) == topk:
                    break
            
            for gr, nu in zip(greed_list, nuc_list):
                if gr not in selected_distractors:
                    selected_distractors.append(gr)
                if nu not in selected_distractors:
                    selected_distractors.append(nu)
                    
                  
        selected_distractors = selected_distractors[:topk]  
        cleaned_inp = "\n"
        for ix, dist in enumerate(selected_distractors):
            cleaned_inp = cleaned_inp + "\n" + str(ix+1) + ". " + dist
        
                
        return cleaned_inp
                



def get_predicted_ids():
    predicted_ids = []
    path = "predictions_mt5/"
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
            out_path = "predictions_mt5/" + qid + ".json"

            dump_json(one_prediction, out_path)


def predict_from_argument(fname):
    
    skipped = 0
    print(f'predicting for subject: {fname}')
       
    with open(fname, "r") as f:
        questions = json.load(f)
        
    mt5_model = Mt5Finetuned(model_path= 'checkpoints/template_based_updated/')

                        
    for question in questions:
                             
        question_text = question["question"].strip()
        answer = question["answer"].strip()
        qid = question["qid"]
        lang = question["language"]
        pred_ids = get_predicted_ids()
                    
        if qid in pred_ids:
            skipped +=1
            continue

        if '###' in answer:
            answers = answer.split('###')
            answers = [ans.strip() for ans in answers]
            answer = random.choice(answers)

            
        inp = mt5_model.build_input(question_text, answer, lang = lang, no_distractors=10, version1=False)
        result = mt5_model.round_robin_selection(inp)
        
        #result = "hi"
        one_prediction = {}
        one_prediction['qid'] = qid
        one_prediction['input'] = inp
        one_prediction['response'] = result
        parent_dir = "predictions_mt5/"
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        out_path = parent_dir + qid + ".json"
        
        print(one_prediction)

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
