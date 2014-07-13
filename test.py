

from recommender import query, search
from recommender.recommender import ArXivRecommender

# Init recommender
recommender = ArXivRecommender('recommender/data/recommender.h5', 'recommender/data/arxiv.db')
doc_id = '1402.1774'
k = 10

import profile

profile.run("query.center(recommender, doc_id, int(k))")
