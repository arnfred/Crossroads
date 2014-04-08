import scipy

import recommender
reload(recommender)
from recommender import Recommender
import util
reload(util)
from util import mystdout

if __name__ == '__main__':
    
    # Set printing config 
    mystdout.setVerbose(True)
    mystdout.setWriteInterval(0.1)

    # Initiliaze the recommender
    db_path = '/Volumes/MyPassport/data/arxiv.db'
    vocab_filename = 'arxiv/voc.txt'
    recommender = Recommender(db_path, vocab_filename)
    K = 400


    # Train the recommender
    # recommender.train(K = K, epochstodo = 2, 
    #     start_date = '2001-01-01 00:00:00.000000',
    #     end_date   = '2014-03-01 00:00:00.000000',
    #     tree_metric = scipy.spatial.distance.cosine)
    # recommender.save('/Volumes/MyPassport/data/recommender_K%d.cpkl'%K)


    recommender.load('/Volumes/MyPassport/data/recommender_K%d.cpkl'%K)
    recommender.open_db_connection(db_path)



    recommender.print_nearest_neighbors(u'1402.1774', 10)


    # D,V = recommender.featureVectors.shape
    # for d in range(D):
    #     minValue = min(recommender.featureVectors[d])
    #     indices = np.where(recommender.featureVectors[d] == minValue)
    #     recommender.featureVectors[d][indices] = 0



    # Slaman's paper: u'1402.1774'
    # Rudiger's papers: u'1401.6060' u'1304.5220' u'1202.4959' u'0901.2370' ...

