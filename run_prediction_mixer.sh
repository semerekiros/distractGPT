#!/bin/bash

#python prediction_mixer.py --pred1 processed/few-shot-newids/english.json --pred2 processed/zero-shot-newids/english.json
#python prediction_mixer.py --pred1 processed/few-shot-newids/french.json --pred2 processed/zero-shot-newids/french.json
#python prediction_mixer.py --pred1 processed/few-shot-newids/biology.json --pred2 processed/zero-shot-newids/biology.json
#python prediction_mixer.py --pred1 processed/few-shot-newids/geography.json --pred2 processed/zero-shot-newids/geography.json
#python prediction_mixer.py --pred1 processed/few-shot-newids/history.json --pred2 processed/zero-shot-newids/history.json
#python prediction_mixer.py --pred1 processed/few-shot-newids/naturalsciences.json --pred2 processed/zero-shot-newids/naturalsciences.json


python prediction_mixer.py --pred1 processed/few-shot-newids/english.json --pred2 processed/zero-shot-newids/english.json --pred3 processed/mt5-newids/english.json
python prediction_mixer.py --pred1 processed/few-shot-newids/french.json --pred2 processed/zero-shot-newids/french.json --pred3 processed/mt5-newids/french.json
python prediction_mixer.py --pred1 processed/few-shot-newids/biology.json --pred2 processed/zero-shot-newids/biology.json --pred3 processed/mt5-newids/biology.json
python prediction_mixer.py --pred1 processed/few-shot-newids/geography.json --pred2 processed/zero-shot-newids/geography.json --pred3 processed/mt5-newids/geography.json
python prediction_mixer.py --pred1 processed/few-shot-newids/history.json --pred2 processed/zero-shot-newids/history.json --pred3 processed/mt5-newids/history.json
python prediction_mixer.py --pred1 processed/few-shot-newids/naturalsciences.json --pred2 processed/zero-shot-newids/naturalsciences.json --pred3 processed/mt5-newids/naturalsciences.json