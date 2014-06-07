"""

Script used to train and build an arXiv recommender

"""

from optparse import OptionParser

from recommender.recommender import ArXivRecommender
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

    K = options.K
    batch_size = 256
    addSmoothing = True

    # Files location
    db_path = 'recommender/data/arxiv.db'
    h5file_path = 'recommender/data/new_recommender.h5'
    vocab_filename = 'recommender/data/voc.txt'

    # Init recommender
    print "Initialize the recommender"
    recommender = Recommender(h5file_path, db_path, mode='w',
        start_date = '2000-01-01 00:00:00.000000',
        end_date   = '2014-05-01 00:00:00.000000',
        categories = set(['cs', 'math', 'q-bio', 'stat']))
    # Add the LDA based recommendation method
    recommender.add_recommendation_method('LDABasedRecommendation')

    print "Train LDA..."
    recommender.methods['LDABasedRecommendation'].train(
        K                   = K, 
        vocab_filename      = vocab_filename,
        stopwords_filename  = stopwords_filename,
        batch_size          = batch_size,
        epochs_to_do        = 2,
        addSmoothing        =addSmoothing,
        start_date          = recommender.start_date,
        end_date            = recommender.end_date,
        categories          = recommender.categories)

    print "Build nearest neighbors..."
    recommender.methods['LDABasedRecommendation'].build_nearest_neighbors(k=20, metric='euclidean')


