# DistractGPT

This repository contains the code for EduQG:

[Distractor generation for multiple-choice questions with predictive prompting and large language models](https://arxiv.org/abs/2307.16338)

If you use part of the code/dataset please cite:  

```  
@misc{bitew2023distractor,
      title={Distractor generation for multiple-choice questions with predictive prompting and large language models}, 
      author={Semere Kiros Bitew and Johannes Deleu and Chris Develder and Thomas Demeester},
      year={2023},
      eprint={2307.16338},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

The **test-data** folder contains all the subjects as JSON files. Each subject is a list of questions with the following attributes:
```
 {
        "question": "Hoe loopt een impuls via een motorisch neuron via de ruggenmergzenuw naar een de effector?",
        "answer": "via de ventrale wortel",
        "language": "nl",
        "distractors": [
            "via de dorsale wortel"
        ],
        "qid": "156a5cb0-1a30-458d-9ad1-e19ddeea05f6"
    }

```
### Predictions from the different models ###
predictions-*model_name* contains the predictions from each of the models. The models are zero-shot, mt5, few-shot and few-shot-static

### Pre-requisites ###

> pip install -r requirements.txt 


#### Train mt5 model from scratch ###

<pre> sh run_mt5.sh
</pre>



