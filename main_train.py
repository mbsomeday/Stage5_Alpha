# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from training.training_template import Ped_Classifier
from configs.pedCls_args import TrainArgs


opts = TrainArgs().parse()
ped_cls = Ped_Classifier(opts=opts)
ped_cls.train()













