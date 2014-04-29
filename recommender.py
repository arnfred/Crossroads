import cPickle
import numpy as np
import scipy
import sqlite3
# import shelve

import sklearn.neighbors

import onlineldavb
reload(onlineldavb)
from onlineldavb.myonlineldavb import OnlineLDA
import arxiv
reload(arxiv)
from arxiv import preprocess
import util
reload(util)
from util import mystdout

import pdb


CATEGORIES_SET = set(['math','cs','q-bio','stat'])

class Recommender():
    """
    Content based recommender for arXiv papers based on LDA
    """

    def __init__(self):
        pass

    # ====================================================================================================

    def train(self, K, voc_filename, batch_size=512, epochs_to_do=2,
        start_date='2000-01-01 00:00:00.000000', end_date='2015-01-01 00:00:00.000000',
        categories=CATEGORIES_SET, addSmoothing=True):
        """
        Train the recommender based on LDA

        Arguments:
        K : int
            Number of topics
        voc_filename : string
            Location of vocabulary file
        batch_size : int (default: 512)
            Size of a batch of document per iteration of the algorithm
        epochs_to_do : int (default: 2)
            Number of epochs to do
        start_date : string (default: '2000-01-01 00:00:00.000000')
            Starting date of the papers to process
        end_date : string (default: '2015-01-01 00:00:00.000000')
            Ending date of the papers to process
            Only papers in this range of date will be taken into account
        categories : iterable (default: all categories)
            Categories of papers to process
        """
        # Initialize the parser
        vocab = open(voc_filename, 'r').read().rstrip('\n').split('\n')
        self.parser = preprocess.parser(vocab)
        # Start date of documents
        self.start_date = start_date
        # End date of documents
        self.end_date = end_date
        # Categories pattern used in SQL query
        self.cat_query_condition = self.make_cat_query_condition(categories)
        # Number of topics
        self.K = K
        # Vocabulary size
        self.W = len(self.parser.vocabulary_)
        # Total number of documents
        self.D = self.cursor.execute("""SELECT COUNT(*) FROM Articles
            WHERE updated_at > '%s' AND updated_at < '%s'
            AND (%s) """ % \
            (self.start_date, self.end_date, self.cat_query_condition)).fetchone()[0]
        # Initialize the online VB algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
        self.olda = OnlineLDA(self.parser, self.K, self.D, 1./self.K, 1./self.K, 1024., 0.7)
        # Actually run the algorithm and get the feature vector for each paper
        self._run_onlineldavb(batch_size, epochs_to_do, addSmoothing)

    def _run_onlineldavb(self, batch_size, epochs_to_do, addSmoothing):
        """
        Run the online VB algorithm on the data
        """
        # Initialize utility variables
        self.ids = list()           # A mapping between papers id and vector indices
        self.feature_vectors = []   # Feature vectors for each document

        perplexity = 0.0
        self.perplexity = []

        # Run multiple over each documents
        for epoch in range(epochs_to_do):
            docs_seen = 0               # Number of documents seen up to now in this epoch
            iteration = 1               # Iteration in this epoch

            # Query for all documents we want
            query = self.cursor.execute("""SELECT abstract,id
                FROM Articles
                WHERE updated_at > '%s' AND updated_at < '%s'
                AND (%s)
                ORDER BY updated_at""" % \
                (self.start_date, self.end_date, self.cat_query_condition))

            # Run over the query results and feed them to the model per batch
            while True:
                # Fetch results
                result = query.fetchmany(batch_size)

                # Stop when there is no more documents
                if len(result) == 0:
                    break

                cur_docset = [i for i,j in result] # Documents set
                docs_seen += len(result)

                if epoch < epochs_to_do-1:
                    # Give them to online LDA
                    (gamma, bound) = self.olda.update_lambda(cur_docset)

                    # Compute an estimate of held-out perplexity
                    (wordids, wordcts) = self.parser.parse_doc_list(cur_docset)
                    perwordbound = bound * len(cur_docset) / (self.D * sum(map(sum, wordcts)))

                    perplexity = np.exp(-perwordbound)
                    if iteration%10 == 0:
                        self.perplexity += [perplexity]

                if epoch == epochs_to_do-1:
                    # In the last epoch, do not update the model (only get the feature vectors)
                    gamma = self.olda.update_gamma(cur_docset)

                    # Keep track of the feature vectors for each documment
                    self.ids += [j for i,j in result] # ids corresponding to current documents
                    self.feature_vectors += self.compute_feature_vectors(gamma, addSmoothing=addSmoothing).tolist()

                mystdout.write("Epoch %d: (%d/%d docs seen), perplexity = %.3f" % \
                    (epoch,docs_seen,self.D,perplexity),
                    iteration, np.ceil(float(self.D)/batch_size))
                iteration += 1

        self.feature_vectors = np.array(self.feature_vectors)
        self.ids = np.array(self.ids)
        mystdout.write("Online VB LDA done. perplexity = %.3f" % perplexity, 1,1, ln=1)


    def compute_feature_vectors(self, gamma, addSmoothing=True):
        """
        Get the feature vectors from LDA gamma parameters
        """
        if addSmoothing:
            return (gamma + self.olda._alpha) / (np.tile(gamma.sum(axis=1), \
                (gamma.shape[1],1)).T + gamma.shape[1]*self.olda._alpha)
        else:
            gamma -= self.olda._alpha
            return gamma / np.tile(gamma.sum(axis=1), (gamma.shape[1],1)).T

    def load_feature_vectors(self, filename):
        """
        Load the feature vectors and the mapping between feature vectors indices and
        the documents indices

        Arguments:
        filename : string
            Location of the file
        """
        try:
            with open(filename, 'rb') as f:
                (self.feature_vectors, self.ids) = cPickle.load(f)
        except IOError:
            print "File %s not found" % filename

    def save_feature_vectors(self, filename):
        """
        Save the feature vectors and the mapping between feature vectors indices and
        the documents indices

        Arguments:
        filename : string
            Location of the file
        """
        try:
            with open(filename, 'wb') as f:
                cPickle.dump((self.feature_vectors, self.ids),f)
        except IOError:
            print "File %s not found" % filename

    # ====================================================================================================

    # def build_id_to_title_map(self, filename):
    #     """
    #     Build the Id to Title mapping from the arXiv database
    #     using Shelve lib
    #     (The database connection needs to be opened)

    #     Arguments:
    #     filename : string
    #         Location where the database will be saved
    #     """
    #     try:
    #         result = self.cursor.execute("SELECT id,title FROM Articles").fetchall()
    #         self.id_to_title_map = shelve.open(filename)
    #         self.id_to_title_map.update(dict(result))
    #     except AttributeError:
    #         print "You should first open the database connection"

    # def load_id_to_title_map(self, filename):
    #     """
    #     Load the Id to Title mapping using Shelve lib

    #     Arguments:
    #     filename : string
    #         Location of the file
    #     """
    #     try:
    #         self.id_to_title_map = shelve.open(filename)
    #     except IOError:
    #         print "File %s not found" % filename

    # def save_id_to_title_map(self):
    #     """
    #     Save the Id to Title mapping (using Shelve lib)
    #     """
    #     self.id_to_title_map.close()

    def get_title(self, paper_id):
        """
        Return the title of the paper with paper_id
        """
        return self.cursor.execute("SELECT title FROM Articles WHERE id == ?", (paper_id,)).fetchone()

    def get_data(self, paper_id):
        """
        Return all the data concerning the paper with paper_id in a dictionary where keys are 
        column names and values are the data
        """
        data = self.cursor.execute("SELECT * FROM Articles WHERE id == ?", (paper_id,)).fetchone()
        names = [row[0] for row in self.cursor.description]
        return dict(zip(names,data))

    # ====================================================================================================

    def build_tree(self, metric='manhattan'):
        """
        Build a ball tree to efficiently compare the the feature vectors

        Argument:
        metric : string or callable (default: manhattan)
            A metric used to compare vectors, or a custom function
            Manhattan distance is used by default as it is a good metric to compare
            probability distributions
        """
        self.btree = sklearn.neighbors.BallTree(self.feature_vectors,
            leaf_size=30, metric=metric)

    def load_tree(self, filename):
        """
        Load the tree

        Arguments:
        filename : string
            Location of the file
        """
        try:
            with open(filename, 'rb') as f:
                self.btree = cPickle.load(f)
        except IOError:
            print "File %s not found" % filename

    def save_tree(self, filename):
        """
        Save the tree

        Arguments:
        filename : string
            Location of the file
        """
        try:
            with open(filename, 'wb') as f:
                cPickle.dump(self.btree, f)
        except IOError:
            print "File %s not found" % filename


     # ====================================================================================================

    # ====================================================================================================

    def close_db_connection(self):
        """
        Close database connection (useful to pickle recommender object)
        """
        self.conn.close()
        self.cursor = None

    def open_db_connection(self, db_path):
        """
        Open database connection

        Arguments:
        db_path : string
            Path to arXiv database

        """
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def make_cat_query_condition(self, categories):
        conditions = []
        for cat in categories:
            conditions.append("categories LIKE '%{}.%'".format(cat))
        return ' OR '.join(conditions)

    # ====================================================================================================

    def get_nearest_neighbors(self, paper_id, k):
        try:
            distances, indices = self.btree.query(self.feature_vectors[np.where(self.ids == paper_id)[0][0]], k)
            return distances[0], indices[0]
        except KeyError:
            print "Unknown paper id: %s" % paper_id


    def print_nearest_neighbors(self, paper_id, k):
        try:
            distances, indices = self.get_nearest_neighbors(paper_id, k)
            for i,(ind,dist) in enumerate(zip(indices,distances)):
                title = self.id_to_title_map[self.ids[ind]]
                print "%d)\tdistance: %f\ttitle: %s" % (i, dist, title)
        except KeyError:
            print "Unknown paper id: %s" % paper_id

    def print_topics(self):
        inverse_voc = {v:k for k, v in self.parser.vocabulary_.items()}
        arg_to_voc = np.vectorize(lambda i: inverse_voc[i])
        for k,topic in enumerate(self.olda._lambda):
            print "===== Topic %d ====="%k
            print ', '.join(arg_to_voc( topic.argsort()[::-1] )[:10])
            print ""

