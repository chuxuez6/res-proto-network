# res-proto-network
## Training a residual block prototypical network

### Install dependencies

* This code has been tested on Ubuntu 16.04 with Python 3.6 and PyTorch 0.4.
* Install [PyTorch and torchvision](http://pytorch.org/).
* Install [torchnet](https://github.com/pytorch/tnt) by running `pip install git+https://github.com/pytorch/tnt.git@master`.
* Install the protonets package by running `python setup.py install` or `python setup.py develop`.

### Set up the Omniglot dataset

* Run `sh download_omniglot.sh`.

### Train the model

* Run `python scripts/train/few_shot/run_train.py`. This will run training and place the results into `results`.
  * You can specify a different output directory by passing in the option `--log.exp_dir EXP_DIR`, where `EXP_DIR` is your desired output directory.
  * If you are running on a GPU you can pass in the option `--data.cuda`.
* Re-run in trainval mode `python scripts/train/few_shot/run_trainval.py`. This will save your model into `results/trainval` by default.

### Evaluate

* Run evaluation as: `python scripts/predict/few_shot/run_eval.py --model.model_path results/trainval/best_model.pt`.
### 
* Due to the sheer size of the raw radar data, we have not uploaded it directly to github, so if you need the raw radar data, please contact us at
Email: 183891760@qq.com
