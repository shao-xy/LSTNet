logger_name = "lstnet"

import sys
sys.path.append('..')
from util.model_util import LoadModel, SaveModel, SaveResults, SaveHistory
from util.Msglog import LogInit

from datetime import datetime

from lstnet_util import GetArguments, LSTNetInit
from lstnet_datautil import DataUtil
from lstnet_model import PreSkipTrans, PostSkipTrans, PreARTrans, PostARTrans, LSTNetModel, ModelCompile
from lstnet_plot import AutoCorrelationPlot, PlotHistory, PlotPrediction

import tensorflow as tf
import numpy as np

custom_objects = {
        'PreSkipTrans': PreSkipTrans,
        'PostSkipTrans': PostSkipTrans,
        'PreARTrans': PreARTrans,
        'PostARTrans': PostARTrans
        }

model_path = "save/mds-web-trace"

if __name__ == '__main__':
	lstnet = LoadModel(model_path, custom_objects)

	k = np.random.randint(1, 10, (1, 48, 2258)) / 20
	out = lstnet.predict(k)

	out = out * 20

	print(out)
