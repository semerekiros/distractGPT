import json
import random
import os
import argparse

def args_parser():
    parser = argparse.ArgumentParser(description="Combine predictions from different models to be displayed in the annotation platfrom")


    parser.add_argument("--pred1",
                        dest="pred1",
                        type=str,
                        default="./processed/",
                        help="path to the subject")
    parser.add_argument("--pred2",
                        dest="pred2",
                        type=str,
                        default="./processed/",
                        help="Where to save your outputs")
    parser.add_argument("--pred3",
                        dest="pred3",
                        type=str,
                        default="./processed/",
                        help="Where to save your outputs")
    parser.add_argument("--lang",
                        dest="lang",
                        type=str,
                        default="nl",
                        help="Choose language as nl, en, fr")

    return parser




def dump_json(data, outpath):
    print('Saving to', outpath)
    with open(outpath, 'w') as out:
        json.dump(data, out, indent=4, separators=(',', ': '))

def read_json(target_file):
    with open(target_file, "r") as fp:
        loaded_json = json.load(fp)

    return loaded_json


random.seed(42)


def refiner(proposed_distractors):
    refined = list()

    duplicate_keeper = dict()
    for dist_info in proposed_distractors:
        distractor = dist_info['distractor']
        distid = dist_info['distid']
        if distractor in duplicate_keeper:

            duplicate_keeper[distractor].append(distid)
        else:
            duplicate_keeper[distractor] = [distid]
            refined.append(dist_info)

    # print(proposed_distractors[0].keys())
    # print(duplicate_keeper)
    # print("unshuffled: ", refined)

    random.shuffle(refined)
    # print("shuffled", refined)
    # print("\n\n\n")
    print(len(refined))

    return refined, duplicate_keeper


def combine_preds(*predictions):
    preds_len = len(predictions)
    refined = dict(predictions[0])
    unrefined = dict(predictions[0])
    duplicates = dict()

    # Do asserts to check the predictions are for the same questions

    for index, cont in enumerate(predictions[0]['content']):
        all_proposed_distractors = list()
        all_proposed_distractors.extend(cont['proposed_distractors'])
        #all_proposed_distractors.extend(cont['ground_truth'])           #Stopped HERE .... the Ground truth should be removed from being annotated.
        qid = cont['qid']

        for i in range(1, preds_len):
            # print(i)
            element = predictions[i]['content'][index]['proposed_distractors']
            # print(element)
            all_proposed_distractors.extend(element)

            # print(len(all_proposed_distractors))

        refined_proposed_distractors, duplicate_keeper = refiner(all_proposed_distractors)

        unrefined['content'][index]['proposed_distractors'] = all_proposed_distractors
        refined['content'][index]['proposed_distractors'] = refined_proposed_distractors

        duplicates[qid] = duplicate_keeper

        # print("refined: ", refined_proposed_distractors)
        # print("unrefined: ", all_proposed_distractors)

    return unrefined, refined, duplicates


if __name__ == '__main__':

    args = args_parser()
    args = args.parse_args()

    pred1_path = args.pred1
    pred2_path = args.pred2
    pred3_path = args.pred3

    subject1 = pred1_path.split("/")[-1]
    subject2 = pred2_path.split("/")[-1]
    subject3 = pred3_path.split("/")[-1]
    assert subject1==subject2==subject3, f"the predictions are not for the same subject, {subject1} and {subject2} and {subject3}"

    pred1 = read_json(pred1_path)
    pred2 = read_json(pred2_path)
    pred3 = read_json(pred3_path)

    unrefined_new, refined_new, duplicates_new = combine_preds(pred1, pred2, pred3)

    refined_folder = "processed/refined/"
    refined_path = refined_folder + subject1

    if not os.path.exists(refined_folder):
        os.mkdir(refined_folder)

    unrefined_folder = "processed/unrefined/"
    unrefined_path = unrefined_folder + subject1

    if not os.path.exists(unrefined_folder):
        os.mkdir(unrefined_folder)



    duplicates_suffix = subject1.split(".json")[0]
    duplicates_suffix = duplicates_suffix  + "_duplicates" + ".json"
    duplicates_path = unrefined_folder + duplicates_suffix



    dump_json(refined_new, refined_path)
    dump_json(unrefined_new, unrefined_path)
    dump_json(duplicates_new, duplicates_path)