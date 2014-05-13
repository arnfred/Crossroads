from recommender.recommender import ArXivRecommender
from recommender.recommender import UnknownAuthorException
from recommender import search

from recommender import util
from recommender.arxiv.preprocess import ArXivVectorizer

recommender = ArXivRecommender('recommender/data/recommender.h5', 'recommender/data/arxiv.db')
recommender.load_all()


start_date = '2000-01-01 00:00:00.000000'
end_date   = '2014-05-01 00:00:00.000000'
categories = set(['cs', 'math', 'q-bio', 'stat'])
author_vectorizer, title_vectorizer = search.train_search(recommender, start_date, end_date, categories)
