import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from random import randrange
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nlgeval import NLGEval
from torch import cuda
from parser import args_parser
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from my_dataset import Race_Dataset, Race_Dataset_test
from utils import *

device = 'cuda' if cuda.is_available() else 'cpu'

def validate_decode(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    max_candidate = 6
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            target_list = data['target_distractors']



            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=150,
                num_beams=3,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
                )
            preds = [normalize_text(tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            if _%100==0:
                print(f'Completed {_}')

            transposed_targetlist = list(map(list, zip(*target_list)))   #transpose and change it to list of lists for later automatic evaluation





            predictions.extend(preds)
            actuals.extend(transposed_targetlist)

    return predictions, actuals





if __name__ == '__main__':
    parser = args_parser()
    args = parser.parse_args()
    checkpoint = 'google/mt5-xxl'
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model = model.to(device)

    test_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 0
    }

    test_set = Race_Dataset_test(args.test_path, tokenizer)

    test_loader = DataLoader(test_set, **test_params)


    predictions, actuals = validate_decode(tokenizer, model, device, test_loader)

    evaluations_metrics = calculate_metrics(actuals, predictions)
    print(evaluations_metrics)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv('./models/predictions_all.csv')
    #main()
