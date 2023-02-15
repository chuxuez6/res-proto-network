# -*- coding:utf8 -*-
import argparse
import torch
import sys
sys.path.append(r"/home/public/b509/code/g21/shy/aaa/proto/proto2")
from eval import main
from sklearn.metrics import confusion_matrix
from protonets.models import few_shot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
parser = argparse.ArgumentParser(description='Evaluate few-shot prototypical networks')

#default_model_path = r'C:\Users\Yuki\Desktop\prototypical-networks-master\prototypical-networks-master\results\best_model.pt'
#跑测试集的参数配置
default_model_path = r'/home/public/b509/code/g21/shy/aaa/proto/proto2/scripts/train/few_shot/results/best_model.pt'
parser.add_argument('--model.model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: {:s})".format(default_model_path))

parser.add_argument('--data.test_way', type=int, default=8, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as model's data.test_way (default: 0)")
parser.add_argument('--data.test_shot', type=int, default=1, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as model's data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=65, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as model's data.query (default: 0)")
parser.add_argument('--data.test_episodes', type=int, default=1, metavar='NTEST',
                    help="number of test episodes per epoch (default: 1000)")

args = vars(parser.parse_args())

main(args)
#画混淆矩阵图
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12,8), dpi=600)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 20:
            plt.text(x_val, y_val, "%d" % (c,), color='white', fontsize=14, fontproperties='Time New Roman',va='center', ha='center')
        elif 0<c and c<=20:
            plt.text(x_val, y_val, "%d" % (c,), color='black', fontsize=14, fontproperties='Time New Roman',va='center', ha='center')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=20,fontproperties='Time New Roman')
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90, fontsize=12,fontproperties='Time New Roman')
    plt.yticks(xlocations, classes, fontproperties='Time New Roman',fontsize=12)
    plt.ylabel('Actual label', fontsize=12,fontproperties='Time New Roman')
    plt.xlabel('Predict label', fontsize=12,fontproperties='Time New Roman')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


classes = ['bow', 'boxing', 'call','Fall down','walk','squat','stand','sit']
#对混淆矩阵做平均处理
print(few_shot.sum)
# cm_sum = torch.div(few_shot.sum, 950)



#绘制混淆矩阵图
plot_confusion_matrix(few_shot.sum, 'confusion_matrix.png', title='confusion matrix')
