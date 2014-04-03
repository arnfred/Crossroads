import cPickle
import string
import numpy as np
import getopt
import sys
import random
import time
import re
import pprint
import sqlite3

import sklearn.neighbors
import scipy

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

class Recommender():

    def __init__(self, db_path, vocab_filename):
        """
        Arguments:
        db_path : string
            Path to arXiv database
        vocab_filename : string
            Path to vocabulary file
        """
        # Open database connection
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        vocab = open(vocab_filename, 'r').read().rstrip('\n').split('\n')
        self.parser = preprocess.parser(vocab)

    def train(self, K, batchsize=512, epochstodo=2, 
        start_date='2000-01-01 00:00:00.000000',
        end_date='2015-01-01 00:00:00.000000',
        tree_metric='euclidean'):
        """
        Train the recommender based on LDA

        Arguments:
        K : int
            Number of topics
        batchsize : int
            Size of a batch of document per iteration of the algorithm
        epochstodo : int
            Number of epochs to do
        start_date : string
            Starting date of the papers to process
        end_date : string
            Ending date of the papers to process
            Only papers in this range of date will be taken into account
        """
        self.K = K
        self.W = len(self.parser.vocabulary_)
        self.D = self.cursor.execute("""SELECT COUNT(*) FROM Articles 
            WHERE updated_at > ? AND updated_at < ?""", 
            (start_date, end_date)).fetchone()[0]

        self.start_date = start_date
        self.end_date = end_date

        self.batchsize = batchsize
        self.limit_per_query = batchsize * 10
        self.epochstodo = epochstodo

        # Initialize the online VB algorithm 
        # with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
        self.olda = OnlineLDA(self.parser, self.K, self.D, 1./self.K, 1./self.K, 1024., 0.7)

        # Actually run the algorithm and get the feature vector for each paper
        self.featureVectors, self.idslist = self._run_onlineldavb()

        # Build ball tree
        mystdout.write("Building ball tree...", 0, 1)        
        self.btree = self.build_tree(tree_metric)
        mystdout.write("Building ball tree... done.", 1, 1, ln=1)  

    def _run_onlineldavb(self):
        """
        Run the online VB algorithm on the data
        """       
        # Initialize utility variables
        idslist = list()                # A mapping between papers id and vector indices
        thetas = np.zeros([0,self.K])   # Feature vectors for each document              

        # Run multiple over each documents
        for epoch in range(self.epochstodo):
            last_id = ''
            last_date = self.start_date
            docs_seen = 0               # Number of documents seen up to now
            iteration = 1
            idstmp = set()

            # Query for all documents we want
            query = self.cursor.execute("""SELECT abstract,id,updated_at
                FROM Articles 
                WHERE updated_at > ? AND updated_at < ? 
                ORDER BY updated_at""", 
                (last_date, self.end_date))

            # Run over the query results and feed them to the model per batch
            while True:
                # Fetch results
                result = query.fetchmany(self.batchsize)

                # Stop when there is no more documents
                if len(result) == 0:
                    break

                cur_docset = [i for i,j,k in result]
                cur_ids = [j for i,j,k in result]
                last_date = max([k for i,j,k in result])
                docs_seen += len(result)
 
                # DEBUG
                if len(idstmp.intersection(cur_ids)) > 0:
                    pdb.set_trace()
                idstmp.update(cur_ids)

                # Give them to online LDA
                (gamma, bound) = self.olda.update_lambda(cur_docset)
                
                # Compute an estimate of held-out perplexity
                (wordids, wordcts) = self.parser.parse_doc_list(cur_docset)
                perwordbound = bound * len(cur_docset) / (self.D * sum(map(sum, wordcts)))           

                # In the last epoch, keep track of the topic weights for each documment
                if epoch == self.epochstodo-1:
                    idslist += cur_ids
                    current_thetas = (gamma + self.olda._alpha)/(np.tile(gamma.sum(axis=1), (gamma.shape[1],1)).T + gamma.shape[1]*self.olda._alpha)
                    thetas = np.concatenate((thetas,current_thetas), axis=0)

                mystdout.write("Epoch: %d, (%d docs seen, date: %s), perplexity = %.3f"%(epoch,docs_seen,last_date[:10],np.exp(-perwordbound)), iteration, np.ceil(float(self.D)/self.batchsize))
                iteration += 1
        mystdout.write("Online VB LDA done.", 1,1, ln=1)

        return thetas, np.array(idslist)

    def build_tree(self, metric='jaccard'):
        return sklearn.neighbors.BallTree(self.featureVectors, leaf_size=30, metric=metric)

    def print_nearest_neighbors(self, paper_id, k):
        try:
            distances, indices = self.btree.query(self.featureVectors[np.where(self.idslist == paper_id)[0]], k)
            for i,(ind,dist) in enumerate(zip(indices[0],distances[0])):
                query = self.cursor.execute("SELECT title FROM Articles WHERE id = '%s'" % self.idslist[ind])
                title = None
                title = query.fetchone()
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



if __name__ == '__main__':
    mystdout.setVerbose(True)
    mystdout.setWriteInterval(0.1)

    db_path = '/Volumes/MyPassport/data/arxiv.db'
    vocab_filename = 'arxiv/voc.txt'
    recommender = Recommender(db_path, vocab_filename)

    recommender.train(K = 20, epochstodo = 2, 
        start_date = '2001-01-01 00:00:00.000000',
        end_date   = '2014-03-01 00:00:00.000000',
        tree_metric = scipy.spatial.distance.cosine)

    with open('recommender.cpkl', 'wb') as f:
        cPickle.dump(recommender, f)

    recommender.print_nearest_neighbors(u'1402.1774', 10)

    # Slaman's paper: u'1402.1774'
    # Rudiger's papers: u'1401.6060' u'1304.5220' u'1202.4959' u'0901.2370'

    # with open('data.cpkl', 'wb') as f:
    #     cPickle.dump((thetas, idslist, olda, parser, btree), f)

    # with open('data.cpkl', 'rb') as f:
    #     thetas, idslist, olda, parser, btree = cPickle.load(f)

