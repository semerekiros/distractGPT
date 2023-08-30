from nlgeval import NLGEval
UNIVERSAL_SEP = "///"

def tok_begin(tokenizer):
    if isinstance(tokenizer._bos_token, str):
        return tokenizer._bos_token
    elif isinstance(tokenizer._cls_token, str):
        return tokenizer._cls_token
    return 'cls'


def tok_sep(tokenizer):
    if isinstance(tokenizer._eos_token, str):
        return tokenizer._eos_token
    elif isinstance(tokenizer._sep_token, str):
        return tokenizer._sep_token
    return 'sep'


def tok_mask(tokenizer):
    if isinstance(tokenizer._mask_token, str):
        return tokenizer._mask_token
    elif isinstance(tokenizer.mask_token, str):
        return tokenizer.mask_token
    return 'msk'


def tok_pad(tokenizer):
    if isinstance(tokenizer._pad_token, str):
        return tokenizer._pad_token
    return 'pad'

def exceed_512_(tokenizer, input_seq, maxlen=510):
    #input_seq = input_seq.replace("[MASK]", tok_mask(tokenizer)).replace("[SEP]", tok_sep(tokenizer)).replace("[CLS]", tok_begin(tokenizer))
    sep_split = input_seq.split("[SEP]")
    #ext_seq = [tok_sep(tokenizer)] + tokenizer.tokenize(tok_sep(tokenizer).join(sep_split[1:])) if len(sep_split) > 1  else []

    if len(sep_split) > 2:
        question_seq = "question: " + sep_split[1]
        question_seq_tok = tokenizer.tokenize(question_seq)
        answer_seq = "answer: " + sep_split[2]
        answer_seq_tok = tokenizer.tokenize("answer: " + sep_split[2])
    else:
        question_seq = []
        answer_seq = []

    context_seq = tokenizer.tokenize("context: "+sep_split[0])

    chunked_context = context_seq[:maxlen - len(question_seq_tok + answer_seq_tok)]

    chunked_context_string = tokenizer.convert_tokens_to_string(chunked_context)
    question_context_answer = question_seq + " " + chunked_context_string + " " + answer_seq



    return question_context_answer    #return if context+question+answer > 512 then truncate the last part of context to make it less than 512

def normalize_text(text):


    text = text.replace("[SEP]", " ")

    text = "".join((char if char.isalpha() or char == " " else " " + char + " ") for char in text)  # separate punctuation
    text = ' '.join(text.split()).lower().strip()  # remove extra blank
    return text

def calculate_metrics(references, hypothesis):
    metrics = NLGEval(metrics_to_omit=['METEOR', 'EmbeddingAverageCosineSimilairty', 'SkipThoughtCS', 'VectorExtremaCosineSimilarity',
                                       'GreedyMatchingScore', 'CIDEr'])      #metric from https://github.com/Maluuba/nlg-eval

    result = metrics.compute_metrics(ref_list=list(map(list, zip(*references))), hyp_list=hypothesis)

    #metrics_dict = metrics.compute_individual_metrics(references, hypothesis)  #references = []  and hypothesis = "string"
    #metrics_dict = metrics.compute_metrics(references, hypothesis)       #references = [[],[],[]] and hypothesis = []

    return result

