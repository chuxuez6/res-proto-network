import argparse
import sys
sys.path.append(r"/home/public/b509/code/g21/shy/aaa/proto/proto2")

from trainval import main
#跑验证集的参数设置
parser = argparse.ArgumentParser(description='Re-run prototypical networks training in trainval mode')

parser.add_argument('--model.model_path', type=str, default='results/best_model.pt', metavar='MODELPATH',
                    help="location of pretrained model to retrain in trainval mode")

args = vars(parser.parse_args())

main(args)
