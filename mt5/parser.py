import argparse
def args_parser():
    parser = argparse.ArgumentParser(description="Distractor Classificaiton")

    parser.add_argument("--epochs",
                        dest="epochs",
                        type=int,
                        default=1,
                        help="Number of gradient descent iterations. Default is 200.")
    parser.add_argument("--learning_rate",
                        dest="learning_rate",
                        type=float,
                        default=0.01,
                        help="Gradient descent learning rate. Default is 0.01.")

    parser.add_argument("--linear_dim",
                        dest="linear_dim",
                        type=int,
                        default=700,
                        help="The size of the linear layer to resize bert outputs. Default is 400.")
    parser.add_argument("--batch_size",
                        dest="batch_size",
                        type=int,
                        default=10,
                        help="Batch size")
    parser.add_argument("--train_path",
                        dest="train_path",
                        type=str,
                        default="./test-data/mini_train_manual.json",
                        help="Training file")
    parser.add_argument("--test_path",
                        dest="test_path",
                        type=str,
                        default="./test-data/test_neg.json",
                        help="Training file")
    parser.add_argument("--output_path",
                        dest="output_path",
                        type=str,
                        default="model_output/model.pt",
                        help="Where to save your model - path")
    parser.add_argument("--model",
                        dest="model",
                        type=str,
                        default="checkpoints/",
                        help="Directory of the model to be loaded")

    return parser
