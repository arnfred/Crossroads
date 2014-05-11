"""
Train and build an arXiv recommender
"""

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
    # Take care of command line arguments
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
    h5file_path = 'data/new_recommender.h5'
    voc_filename = 'data/voc.txt'
    id_to_title_map_filename = 'data/id_to_title_map'
    tree_filename = 'data/tree_K%d%s.cpkl' % (K, "" if addSmoothing else "_nosmoothing")
    feature_vectors_filename = 'data/feature_vectors_2013_K%d%s.cpkl' % (K,  "" if addSmoothing else "_nosmoothing")

    # Init recommender
    recommender = Recommender(h5file_path, db_path)
    # Open DB connection
    recommender.open_db_connection()

    # Train the recommender and save the feature vectors
    recommender.train(K = K, 
        voc_filename = voc_filename,
        batch_size = batch_size,
        epochs_to_do = 2, 
        start_date = '2000-01-01 00:00:00.000000',
        end_date   = '2014-05-01 00:00:00.000000',
        categories = set(['cs', 'math', 'q-bio', 'stat']),
        addSmoothing=addSmoothing)
    
    recommender.topics = recommender.olda._lambda
    inverse_voc = {v:k for k, v in recommender.parser.vocabulary_.items()}
    recommender.vocabulary = inverse_voc.values()

    print "Save recommender..."
    recommender.save("topics")
    recommender.save("vocabulary")
    recommender.save("ids")
    recommender.save("feature_vectors")
    
    print "Build tree..."
    recommender.build_tree("euclidean")

    print "Build nearest neighbors..."
    recommender.build_nearest_neighbors(10)


    # Salman's paper: u'1402.1774'
    # Rudiger's papers: u'1401.6060' u'1304.5220' u'1202.4959' u'0901.2370' ...


