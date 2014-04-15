import scipy
import numpy as np
from optparse import OptionParser

import recommender
reload(recommender)
from recommender import Recommender
import util
reload(util)
from util import mystdout

if __name__ == '__main__':
    # === Take care of command line arguments
    parser = OptionParser()        
    parser.add_option("-K", type="int", dest="K", default=200,
        help="Number of topics")
    (options, args) = parser.parse_args()
    
    # Set printing config 
    mystdout.setVerbose(True)
    mystdout.setWriteInterval(0.1)

    K = options.K
    batch_size = 256
    addSmoothing = True

    # Files location
    db_path = 'data/arxiv.db'
    voc_filename = 'data/voc.txt'
    id_to_title_map_filename = 'data/id_to_title_map'
    tree_filename = 'data/tree_K%d%s.cpkl' % (K, "" if addSmoothing else "_nosmoothing")
    feature_vectors_filename = 'data/feature_vectors_2013_K%d%s.cpkl' % (K,  "" if addSmoothing else "_nosmoothing")

    # Init recommender
    recommender = Recommender()
    
    # Open DB connection
    recommender.open_db_connection(db_path)


    # Train the recommender and save the feature vectors
    recommender.train(K = K, 
        voc_filename = voc_filename,
        batch_size = batch_size,
        epochs_to_do = 2, 
        start_date = '2013-01-01 00:00:00.000000',
        end_date   = '2014-03-01 00:00:00.000000',
        categories = set(['cs', 'math']),
        addSmoothing=addSmoothing)
    recommender.save_feature_vectors(feature_vectors_filename)


    # print "Load feature vectors..."
    # recommender.load_feature_vectors(feature_vectors_filename)
    

    # === Build id to title map and save it
    print "Load id to tile map..."
    recommender.load_id_to_title_map(id_to_title_map_filename)
    # === Build the tree and save it
    mystdout.write("Build tree...", 0, 1)
    recommender.build_tree(metric='euclidean')
    mystdout.write("Build tree... done.", 1, 1, ln=1)
    recommender.save_tree(tree_filename)




    # D,V = recommender.featureVectors.shape
    # for d in range(D):
    #     minValue = min(recommender.featureVectors[d])
    #     indices = np.where(recommender.featureVectors[d] == minValue)
    #     recommender.featureVectors[d][indices] = 0



    # Salman's paper: u'1402.1774'
    # Rudiger's papers: u'1401.6060' u'1304.5220' u'1202.4959' u'0901.2370' ...




    inverse_voc = {v:k for k, v in recommender.parser.vocabulary_.items()}
    arg_to_voc = np.vectorize(lambda i: inverse_voc[i])
    for k in range(recommender.olda._lambda.shape[0]):
        print "===== Topic %d ====="%k
        print arg_to_voc( recommender.olda._lambda[k].argsort()[::-1] )[:20]
        print ""



    # Query for one paper
    recommender.print_nearest_neighbors(u'1402.1774', 10)


