#!/usr/bin/env python3

logger_name = "lstnet"

import sys
sys.path.append('/home/ceph/LSTNet/')
from util.model_util import LoadModel, SaveModel, SaveResults, SaveHistory
from util.Msglog import LogInit

from datetime import datetime

from lstnet_util import GetArguments, LSTNetInit
from lstnet_datautil import DataUtil
from lstnet_model import PreSkipTrans, PostSkipTrans, PreARTrans, PostARTrans, LSTNetModel, ModelCompile
from lstnet_plot import AutoCorrelationPlot, PlotHistory, PlotPrediction

import tensorflow as tf
import numpy as np

import pathlib

custom_objects = {
        'PreSkipTrans': PreSkipTrans,
        'PostSkipTrans': PostSkipTrans,
        'PreARTrans': PreARTrans,
        'PostARTrans': PostARTrans
        }

model_path = str(pathlib.Path(__file__).parent.resolve()) + "/save/mds-web-trace"

class IndexedChunk:
	cur_start = 0

	def __init__(self, chunk, cur_width):
		self._rawchunk = chunk
		self._chunk = np.zeros(chunk.shape)
		self._pos = (IndexedChunk.cur_start, IndexedChunk.cur_start + cur_width)
		chunk_width = chunk.shape[1]
		scale_factor = np.ones(chunk_width)
		for i in range(chunk_width):
			scale_factor[i] = np.max(np.abs(chunk[:, i]))
			self._chunk[:, i] = chunk[:, i] / scale_factor[i]
		self._scale = scale_factor[:cur_width]
		IndexedChunk.cur_start += cur_width

	def __str__(self):
		s = 'IndexedChunk(\n\t'
		_len = self._pos[1] - self._pos[0]
		s += str(self._chunk[:, :(_len)])
		s += '\n\t'
		s += str(self._pos)
		s += '\n\t'
		s += str(self._scale[:_len].tolist())
		return s

	@staticmethod
	def reset():
		IndexedChunk.cur_start = 0

class LSTNetPredictor:
	def __init__(self):
		self.lstnet = LoadModel(model_path, custom_objects)
		_, self.model_row, self.model_col = self.lstnet.input_shape

	def fit(self, load_matrix):
		# Suppose input matrix is M * N (we assume its dimension is 2)
		load_matrix = np.array(load_matrix).transpose()
		assert(load_matrix.ndim == 2)
		raw_h, raw_w = load_matrix.shape

		if raw_h > self.model_row:
			# disregard additional time epochs (if M > model_row => model_row * N)
			load_matrix = load_matrix[:self.model_row]
		elif raw_h < self.model_row:
			# fill matrix with "empty" (random small) lines
			dull_patching_matrix = np.random.randint(1, 10, (self.model_row - raw_h, raw_w))
			load_matrix = np.concatenate((dull_patching_matrix, load_matrix), axis=0)

		# Now matrix should be model_row * N
		# let's (maybe) split and/or expand these matrixes
		nchunks = np.ceil(raw_w / self.model_col).astype(np.int64).item()
		assert(type(nchunks) == int)

		IndexedChunk.reset()
		fitted_chunks = []
		for chunk in np.array_split(load_matrix, nchunks, axis=1):
			c_h, c_w = chunk.shape
			dull_patching_matrix = np.random.randint(1, 10, (c_h, self.model_col - c_w))
			fitted_chunks.append(IndexedChunk(np.concatenate((chunk, dull_patching_matrix), axis=1), c_w))
		return fitted_chunks

	def predict(self, load_matrix):
		#out = lstnet.predict(self.fit(load_matrix))
		final = []
		for ichunk in self.fit(load_matrix):
			#print(ichunk)
			out = self.lstnet.predict(np.array([ichunk._chunk]))
			start, end = ichunk._pos
			#print(out[0])
			final[start:end] = (out[0][start:end] * ichunk._scale).tolist()

		return final

if __name__ == '__main__':
	pred = LSTNetPredictor()
	load_matrix = [[100, 200], [300, 400], [500, 600]]

	print()
	print(pred.predict(load_matrix))

	sys.exit(0)

	lstnet = LoadModel(model_path, custom_objects)

	#k = np.random.randint(1, 10, (1, 48, 2258)) / 20
	k = np.random.randint(1, 10, (1, 48, 2258))
	out = lstnet.predict(k)

	#out = out * 20

	print(out.tolist()[0])
	#print(len(out.tolist()[0]))
