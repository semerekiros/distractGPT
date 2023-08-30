import argparse

def args_parser():
    parser = argparse.ArgumentParser(description="Extract test-data from televic dump")



    parser.add_argument("--subject",
                        dest="subject",
                        type=str,
                        default="./test-data/biology.json",
                        help="path to the subject")
    parser.add_argument("--output_path",
                        dest="output_path",
                        type=str,
                        default="predictions-few-shot/",
                        help="Where to save your outputs")
    parser.add_argument("--lang",
                        dest="lang",
                        type=str,
                        default="nl",
                        help="Choose language as nl, en, fr")

    return parser

