import argparse
import torch
import sys
sys.path.append(r"/home/public/b509/code/g21/shy/aaa/proto/proto2")
from test_single import main
from sklearn.metrics import confusion_matrix
from protonets.models import few_shot
import matplotlib.pyplot as plt
import numpy as np
parser = argparse.ArgumentParser(description='Evaluate single image')

#default_model_path = r'C:\Users\Yuki\Desktop\prototypical-networks-master\prototypical-networks-master\results\best_model.pt'

default_model_path = r'/home/public/b509/code/g21/shy/aaa/proto/proto2/scripts/train/few_shot/results/best_model.pt'
parser.add_argument('--model.model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: {:s})".format(default_model_path))
parser.add_argument('--data.test_way', type=int, default=7, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as model's data.test_way (default: 0)")
parser.add_argument('--data.test_shot', type=int, default=5, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as model's data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=10, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as model's data.query (default: 0)")
parser.add_argument('--data.test_episodes', type=int, default=10, metavar='NTEST',
                    help="number of test episodes per epoch (default: 1000)")
args = vars(parser.parse_args())

main(args)