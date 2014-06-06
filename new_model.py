import numpy as np
from scipy import sparse
import time

from sklearn.preprocessing import normalize

import recommender
reload(recommender)
from recommender.recommender import ArXivRecommender
from recommender import util

from recommender.recommendation_technics import *

# Init recommender
def init_recommender() :
	recommender = ArXivRecommender('recommender/data/recommender.h5', 'recommender/data/arxiv.db', mode='a')
	recommender.load_all()
	return recommender
recommender = init_recommender()

recommender.author_recommendation = LDABasedRecommendation()
