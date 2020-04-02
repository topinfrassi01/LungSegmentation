from __future__ import division
import json
import os
import shutil

from model import SegmentModel
from preprocess import GlandHandler
import sys
from metrics import get_scores

import numpy as np


__author__ = "Mathias Baltzersen and Rasmus Hvingelby"

data_path = sys.argv[1]

hps = {"bs": 1,
       "epochs": 10,  # 400
       "dropout_prob": 0.5,
       "acquisition": "variance",  # choose from 'variance', 'entropy', 'KL_divergence' and 'random'
       "exp_name": "my_testing",  
       "ensemble_method": "dropout",  # choose from 'dropout' and 'bootstrap'
       "lr": 0.0005,
       "img_size": 256,
       "classes": 3,
       "scale_nc": 1, # The scaling factor for number of channels. Paper does not scale but their code scales with 2
       "contour_loss_weight": 1.0,
       "l2_scale": 0.0
       }

path = "./" + hps.get("exp_name")
paths = [path + "/model_" + str(i) for i in range (4)] if hps.get("ensemble_method")=='bootstrap' else []
paths.insert(0, path)
hps_file = "/hps.json"

for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)
    else:
        shutil.rmtree(p)
        os.makedirs(p)

with open(path + hps_file, 'w') as file:
    json.dump(hps, file)

x_train, y_train_seg, y_train_cont, x_a_test, y_a_test, x_b_test, y_b_test = GlandHandler(data_path).get_gland()

total_num_train_images = x_train.shape[0]

net = SegmentModel(hps)

net.train(x_train, y_train_seg, y_train_cont)
results_a = net.evaluate(x_a_test, y_a_test)
results_b = net.evaluate(x_b_test, y_b_test)

final_pred_a = net.final_predictions(results_a)
final_pred_b = net.final_predictions(results_b)

get_scores(final_pred_a, y_a_test, "test_a", hps=hps)
get_scores(final_pred_b, y_b_test, "test_b", hps=hps)