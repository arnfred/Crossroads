import recommender
reload(recommender)
from recommender import query, search
from recommender.recommender import ArXivRecommender
import numpy as np

# Init recommender
def init_recommender() :
    recommender = ArXivRecommender('recommender/data/recommender.h5', 'recommender/data/arxiv.db')
    recommender.load_all()
    return recommender
recommender = init_recommender()


self = recommender
idx = np.where(np.array(self.ids[:]) == '1402.1774')[0][0]

percentile = 0.8
others_top = np.sum( 
	np.cumsum(np.sort(self.feature_vectors[:], axis=1)[:,::-1], axis=1) < percentile,
	axis=1)
others = np.argsort(self.feature_vectors[:], axis=1)[:,::-1]