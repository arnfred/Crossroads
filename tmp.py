import scipy.sparse as scsp
import numpy as np

import recommender
reload(recommender)
from recommender import query, search
from recommender.recommender import ArXivRecommender



# Init recommender
def init_recommender() :
    recommender = ArXivRecommender('recommender/data/recommender.h5', 'recommender/data/arxiv.db')
    recommender.load_all()
    return recommender
recommender = init_recommender()


self = recommender
paper_id = '1402.1774'
k = 40
percentile = 0.8

distances, indices = recommender.get_nearest_neighbors_online(paper_id, k, percentile=percentile)