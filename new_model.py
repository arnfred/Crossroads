import numpy as np
from scipy import sparse
import time

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

import recommender
reload(recommender)
from recommender.recommender import ArXivRecommender
from recommender import util

from recommender.recommendation_methods import LDABasedRecommendation, AuthorBasedRecommendation

from recommender import search

h5file_path = 'recommender/data/recommender.h5'
db_path = 'recommender/data/arxiv.db'

recommender = ArXivRecommender(h5file_path, db_path, mode='r')
self = recommender

paper_id = '1402.1774'
k = 30
distances, indices, method_dist, method_weight = recommender.get_nearest_neighbors_online(paper_id, k)

print "rank \t| distance | id%s| LDA%s| Authors" % (" "*16, " "*4)
for rank,(idx,dist) in enumerate(zip(indices,distances)):
	doc_id = self.ids[idx]
	print "%d\t| %.4f   | %s%s| %.4f\t| %.4f" % (rank, dist, doc_id, " "*(18-len(doc_id)), method_dist['LDABasedRecommendation'][idx], method_dist['AuthorBasedRecommendation'][idx])
