import numpy as np
import scipy.sparse as scsp
import sqlite3
import cPickle
import tables

import sklearn.neighbors

from onlineldavb.myonlineldavb import OnlineLDA
from arxiv.preprocess import ArticleParser
import util
from util import mystdout


class ArXivRecommender():
    """
    Content based recommender for arXiv papers based on LDA
    """

    def __init__(self, hdf5_path, db_path, mode="r"):
        """

        Arguments:
        hdf5_path : str
            Location of the hdf5 file. If it does not exist, it is created
        db_path : str
            Location of the db file
        """
        self.h5file = tables.open_file(hdf5_path, mode=mode, title="Trailhead - arXiv recommender")
        self.db_path = db_path

    # ====================================================================================================

    def train(self, K, vocab_filename, stopwords_filename, batch_size=512, epochs_to_do=2,
        start_date='2000-01-01 00:00:00.000000', end_date='2015-01-01 00:00:00.000000',
        categories=set(), addSmoothing=True):
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
        # Open db connection
        self.open_db_connection()
        # Initialize the parser
        vocab = open(vocab_filename, 'r').read().rstrip('\n').split('\n')
        stopwords  = open(stopwords_filename, 'ru').read().split(',')
        self.parser = ArticleParser(vocab)
        # Start date of documents
        self.start_date = start_date
        # End date of documents
        self.end_date = end_date
        # Categories pattern used in SQL query
        self.cat_query_condition = util.make_cat_query_condition(categories)
        # Number of topics
        self.K = K
        # Vocabulary size
        self.W = len(self.parser.vocabulary_)
        # Total number of documents
        self.D = self.cursor.execute("""SELECT COUNT(*) FROM Articles
            WHERE updated_at > '%s' AND updated_at < '%s'
            %s """ % \
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
                %s
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
                    self.ids += [str(j) for i,j in result] # ids corresponding to current documents
                    self.feature_vectors += self.compute_feature_vectors(gamma, addSmoothing=addSmoothing).tolist()

                mystdout.write("Epoch %d: (%d/%d docs seen), perplexity = %.3f" % \
                    (epoch,docs_seen,self.D,perplexity),
                    iteration, np.ceil(float(self.D)/batch_size))
                iteration += 1

        # Convert the lists to a Numpy arrays
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

    # ====================================================================================================

    def get_title(self, paper_id):
        """
        Return the title of the paper with paper_id
        """
        self.open_db_connection()
        try:
            title = self.cursor.execute("SELECT title FROM Articles WHERE id == ?", (paper_id,)).fetchone()[0]
        except IndexError:
            raise UnknownIDException(paper_id)

    def get_data(self, paper_id):
        """
        Return all the data concerning the paper with paper_id in a dictionary where keys are
        column names and values are the data
        """
        try:
            data = self.cursor.execute("SELECT * FROM Articles WHERE id == ?", (paper_id,)).fetchone()
            names = [row[0] for row in self.cursor.description]
            return dict(zip(names,data))
        except TypeError:
            raise UnknownIDException(paper_id)

    def get_papers_from_author(self, author):
        self.open_db_connection()
        formatted_author = "%{0}%".format(author)
        ids = self.cursor.execute("SELECT id FROM Articles WHERE authors LIKE ?", (formatted_author,)).fetchall()
        if len(ids) == 0:
            raise UnknownAuthorException(author)



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

    def build_nearest_neighbors(self, k):
        """
        Build the matrix a k nearest neighbors for every paper in the tree
        """

        N,K = self.feature_vectors[:,:].shape
        batch_size = 100

        try:
            self.h5file.createCArray("/", "neighbors_distances", tables.Float64Atom(), shape=(N,k))
            self.h5file.createCArray("/", "neighbors_indices", tables.UInt64Atom(), shape=(N,k))
            self.neighbors_distances = self.h5file.root.neighbors_distances
            self.neighbors_indices = self.h5file.root.neighbors_indices
        except Exception:
            self.load("neighbors_distances")
            self.load("neighbors_indices")

        for i in range(N/batch_size):
            mystdout.write("Query knn... %d/%d"%(i*batch_size,N), i*batch_size,N)
            idx = range(i*batch_size, (i+1)*batch_size)
            distances, indices = self.btree.query(self.feature_vectors[idx,:], k=k+1)
            self.neighbors_distances[idx,:] = distances[:,1:]
            self.neighbors_indices[idx,:] = indices[:,1:]

        mystdout.write("Query knn... %d/%d"%(i*batch_size,N), i*batch_size,N, ln=1)

    # ====================================================================================================

    def close_db_connection(self):
        """
        Close database connection
        """
        self.conn.close()
        self.cursor = None

    def open_db_connection(self):
        """
        Open database connection
        """
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()


    # ====================================================================================================

    def load_all(self):
        """
        Load everything we need
        """
        group = self.h5file.root.recommender
        for array in self.h5file.list_nodes(group):
            self.load(array.name)

    def load(self, attr):
        """
        Load an attribute from hdf5 file
        """
        setattr(self, attr, getattr(self.h5file.root.recommender, attr))

    def save(self, attr):
        """
        Save an attribute to the hdf5 file
        """
        try:
            self.h5file.createArray(self.h5file.root.recommender, attr, getattr(self, attr))
        except tables.exceptions.NodeError:
            # If the array already exists, update it
            getattr(self.h5file.root, attr)[:] = getattr(self, attr)

    # ====================================================================================================

    def get_nearest_neighbors(self, paper_id, k):
        """
        Get the k nearest neighbors for a given paper_id

        Arguments:
        paper_id : int
            Id of the paper
        k : int
            Number of neighbors to return
        """
        try:
            idx = np.where(np.array(self.ids[:]) == paper_id)[0][0]
            distances = self.neighbors_distances[idx,:k]
            indices = self.neighbors_indices[idx,:k]
            return distances, indices
        except KeyError:
            print "Unknown paper id: %s" % paper_id


    def get_nearest_neighbors_online(self, paper_id, k, metric='jaccard-euclidean', percentile=1.0):
        """
        Get the k nearest neighbors for a given paper_id by computing them online
        based on some metric
        """
        try:
            # Id of paper in feature vector matrix
            idx = np.where(np.array(self.ids[:]) == paper_id)[0][0]
            # Top topics for this paper
            topics = self.get_top_topics(idx, float(percentile))
            N = len(topics)

            this_jaccard = scsp.coo_matrix(
                (
                    np.ones(N),
                    (
                        np.zeros(N),
                        topics
                    ),
                ), shape=[1, self.feature_vectors.shape[1]])

            # Top topics of all papers
            others_top = np.sum( 
                np.cumsum(np.sort(self.feature_vectors[:], axis=1)[:,::-1], axis=1) < percentile,
                axis=1)
        
        except KeyError:
            print "Unknown paper id: %s" % paper_id

    # ====================================================================================================

    def get_topic_top_words(self, topic_id, k):
        """
        Get the top k words for a given topic (i.e. the ones with highest
        probability)

        Arguments:
        topic_id : int
            Id of the topic
        k : int
            Number of words to return
        """
        return self.vocabulary[np.argsort(self.topics[topic_id][::-1])][:k]

    def get_top_topics(self, idx, k):
        """
        Get the top k topics for a given paper (i.e. the ones with highest
        probability)

        Arguments:
        idx : int
            Id of the paper
        k : int or fload
            if a int is provided: Number of topics to return
            if a float is provided: portion of topics (in probability) to return
        """
        if type(k) is int:
            return np.argsort(self.feature_vectors[idx])[::-1][:k]
        elif type(k) is float:
            n = sum( np.cumsum(np.sort(self.feature_vectors[idx])[::-1]) < k)
            return np.argsort(self.feature_vectors[idx])[::-1][:n]


# ========================================================================================================

class UnknownIDException(Exception):
    """
    Exception raised if a unknown paper id is queried
    """
    def __init__(self, paper_id):
        self.paper_id = paper_id

    def __str__(self):
        return repr("UnknownIDException: Unknown paper id %s" % self.paper_id)

class UnknownAuthorException(Exception):
    """
    Exception raised if a unknown paper id is queried
    """
    def __init__(self, author):
        self.paper_id = author

    def __str__(self):
        return repr("UnknownAuthorException: Unknown author %s" % self.author)
