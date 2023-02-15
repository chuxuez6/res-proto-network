import os
import json
import math
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(r"/home/public/b509/code/g21/shy/aaa/proto/proto2/protonets")
import torch
import torchnet as tnt

from torch.autograd import Variable
from functools import partial
from protonets.models import few_shot
from protonets.models import register_model


from torch import nn
from PIL import Image
from torchvision import transforms,datasets
from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils_1
import protonets.utils.model as model_utils_1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_image_path(key, out_field, d):
    d[out_field] = Image.open(d[key])
    return d

def convert_tensor(d):
    d = 1.0 - torch.from_numpy(np.array(d, np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d.size[0], d.size[1])
    return d

def rotate_image(rot, d):
    d = d.rotate(rot)
    return d

def scale_image(height, width, d):
    d = d.resize((height, width))
    return d

global img_
img_path = r'/home/public/b509/code/g21/shy/aaa/proto/proto2/data/omniglot/dataset/dataset/test_bike/character01/bike_7.png'
rot = ['0', '90', '180', '270']
for i in range(0,3):
    transform_valid = transforms.Compose([
#        transforms.Resize((56, 56), interpolation=2),
#        transforms.ToTensor()
#        partial(load_image_path, 'file_name', 'data'),
        partial(rotate_image,  float(rot[i])),
        partial(scale_image, 56, 56),
        partial(convert_tensor)]
    )
    img = Image.open(img_path)
    img_ = transform_valid(img).unsqueeze(0)  # 拓展维度
img_ = img_.to(device)
img_ = Variable(img_)
test_img = img_
def main(opt):
    # load model
    model = torch.load(opt['model.model_path'])
    model.to(device)
    model.eval()

    class_names = ['person', 'bike', 'car', 'motorbike']  # 这个顺序很重要，要和训练时候的类名顺序一致

    print(img_)
    #    test_img = few_shot.load_protonet_conv.encoder1.forward(img_)
    #    x_img = few_shot.encoder_outer.forward(img_)
    #    print(test_img)
    #    print(protonets.models.few_shot.z_proto)
    # load opts
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # Postprocess arguments
    model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')
    # construct data
    data_opt = {'data.' + k: v for k, v in filter_opt(model_opt, 'data').items()}
    print(data_opt)
    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k, v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'],
        data_opt['data.test_query'], data_opt['data.test_episodes']))

    torch.manual_seed(1234)
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    data = data_utils_1.load(data_opt, ['test'])

    if data_opt['data.cuda']:
        model.cuda()

    meters = {field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields']}

    model_utils_1.evaluate(model, data['test'], meters, desc="test")

    for field, meter in meters.items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean,1.96 * std / math.sqrt(data_opt['data.test_episodes'])))

#    print(model(img_))
#    outputs = model(img_)
    print(few_shot.test_hat)
    result = class_names[few_shot.test_hat[0]]
    print('predicted:', result)
