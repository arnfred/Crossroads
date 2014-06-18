import numpy as np
from scipy import sparse
import time
import tables

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

import recommender
reload(recommender)
from recommender.recommender import ArXivRecommender
from recommender import util
from recommender.util import mystdout

from recommender.recommendation_methods import LDABasedRecommendation, AuthorBasedRecommendation
from recommender import search



h5file_path = 'recommender/data/recommender.h5'
db_path = 'recommender/data/arxiv.db'

recommender = ArXivRecommender(h5file_path, db_path, mode='a')
self = recommender.methods['AuthorBasedRecommendation']

N = self.feature_vectors.shape[0]
batch_size = 1000
for i in np.arange(np.ceil(N/batch_size)):
	mystdout.write("Query nearest neighbors... %d/%d"%(i*batch_size,N), i*batch_size,N)
	idx = np.arange(i*batch_size, (i+1)*batch_size)
	self.neighbors_distances[idx,:] = 1 - self.neighbors_distances[idx,:]
	self.h5file.flush()
