"""

Script used to train and build an arXivRecommender

WARNING: The script takes a very long time to run !

"""

from optparse import OptionParser

from recommender.recommender import ArXivRecommender
from recommender import util
from recommender.util import mystdout

if __name__ == '__main__':
    # Take care of command line arguments
    parser = OptionParser()        
    parser.add_option("-K", type="int", dest="K", default=200,
        help="Number of topics")
    (options, args) = parser.parse_args()
    
    # Set printing config 
    mystdout.setVerbose(True)
    mystdout.setWriteInterval(0.1)

    # Files location
    db_path = 'recommender/data/arxiv.db'
    h5file_path = 'recommender/data/new_recommender2.h5'
    vocab_filename = 'recommender/data/voc.txt'

    # Init recommender
    print "Initialize the recommender"
    recommender = ArXivRecommender(h5file_path, db_path, mode='w',
        start_date = '2000-01-01 00:00:00.000000',
        end_date   = '2014-05-01 00:00:00.000000',
        categories = set(['cs', 'math', 'q-bio', 'stat']))
   
    # Add the LDA based recommendation method
    recommender.add_recommendation_method('LDABasedRecommendation')

    # Train the recommendation methods (compute the feature vectors for each article, but does not store them on the HDF5 file)
    print "Train LDA..."
    recommender.methods['LDABasedRecommendation'].train(
        K                   = options.K, 
        vocab_filename      = vocab_filename,
        batch_size          = 256,
        epochs_to_do        = 2)
    
    # Compute the nearest neighbors based on these feature vectors
    print "Build nearest neighbors..."
    recommender.methods['LDABasedRecommendation'].build_nearest_neighbors(k=30, metric='total-variation')


