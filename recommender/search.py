"""

Module for searching for papers given a term

"""

import json
import sqlite3
import numpy as np
from scipy import sparse
import tables
import time

import query
from recommender import UnknownIDException
import util
from arxiv.preprocess import SearchVectorizer
from util import mystdout


def train_search(recommender, start_date, end_date, categories):
    """
    Train the search engine for a given recommender and save search data into 
    its hdf5 file
    """
    author_vectorizer = SearchVectorizer(category = 'author', training=True)
    title_vectorizer = SearchVectorizer(category = 'title', training=True) 

    # Categories pattern used in SQL query
    cat_query_condition = util.make_cat_query_condition(categories)
    # Query for all documents we want
    recommender.open_db_connection()
    # Fetch all data
    sql_query_string = """SELECT id,authors,title
        FROM Articles
        WHERE updated_at > '%s' AND updated_at < '%s'
        %s
        ORDER BY updated_at""" % \
        (start_date, end_date, cat_query_condition)

    print "Execute SQL query..."
    result = recommender.cursor.execute(sql_query_string).fetchall()

    ids = [e[0] for e in result]
    authors = [e[1] for e in result]
    titles = [e[2] for e in result]

    # Check that that both the recommender has ids in the same order
    assert all(ids == recommender.ids[:]), "Ids list are different"

    # Fit and transform authors
    print "Compute author feature vectors..."
    author_matrix = author_vectorizer.fit_transform(authors)
    author_matrix = sparse.csr_matrix(author_matrix)

    print "Compute title feature vectors..."
    title_matrix = title_vectorizer.fit_transform(titles)
    title_matrix = sparse.csr_matrix(title_matrix)

    print "Save search engin data..."
    # Create search group if it does not already exists
    try:
        recommender.h5file.create_group("/", 'search', 'Search engine data')
    except tables.exceptions.NodeError:
        pass
    # Store author and title sparse matrices
    util.store_sparse_mat(author_matrix, 'author_matrix', recommender.h5file, recommender.h5file.root.search)
    util.store_sparse_mat(title_matrix, 'title_matrix', recommender.h5file, recommender.h5file.root.search)

    # Store author names list
    inverse_voc = {v:k for k, v in author_vectorizer.vocabulary_.items()}
    author_list = inverse_voc.values()
    author_list = np.array(author_list)
    util.store_carray(author_list, 'author_list', recommender.h5file, recommender.h5file.root.search)

    # Store title vocabulary list
    inverse_voc = {v:k for k, v in title_vectorizer.vocabulary_.items()}
    title_list = inverse_voc.values()
    title_list = np.array(title_list)
    util.store_carray(title_list, 'title_list', recommender.h5file, recommender.h5file.root.search)

    return author_vectorizer, title_vectorizer



class ArXivSearchEngine(object):

    def __init__(self, recommender):
        self.recommender = recommender

        # === Load author search engine
        # Load author matrix
        self.author_matrix = util.load_sparse_mat('author_matrix', recommender.h5file, recommender.h5file.root.search)
        # Initialize author vectorizer
        self.author_vectorizer = SearchVectorizer(
            category = 'author',
            vocabulary = recommender.h5file.root.search.author_list[:])

        # === Load title search engine
        # Load title matrix
        self.title_matrix = util.load_sparse_mat('title_matrix', recommender.h5file, recommender.h5file.root.search)
        # Initialize title vectorizer
        self.title_vectorizer = SearchVectorizer(
            category = 'title',
            vocabulary = recommender.h5file.root.search.title_list[:])

    def search_per_author(self, search_input):
        # Buil author query vector
        q = self.author_vectorizer.transform((search_input,))
        if q.sum() == 0:
            return set()
        # Build recommendation
        r = q * self.author_matrix.T
        r = r.toarray()[0]
        if r.sum() > 0:
            r_args = np.argsort(r)[::-1]
            r = np.sort(r)[::-1]
            r_args = r_args[r == r.max()]
            return set(self.recommender.ids[r_args])
        return set()
        
    def search_per_title(self, search_input):
        # Buil title query vector
        q = self.title_vectorizer.transform((search_input,))
        if q.sum() == 0:
            return set()
        # Build recommendation
        r = q * self.title_matrix.T
        r = r.toarray()[0]
        if r.sum() > 0:
            r_args = np.argsort(r)[::-1]
            r = np.sort(r)[::-1]
            r_args = r_args[r > 0]
            return set(self.recommender.ids[r_args])
        return set()

    def query(self, search_input) :

        """ Searches the arXiv papers to see if the search input matches authors,
        titles or ids. A list is returned with the results. For example:
            [{ "id" : "dsfgsd.dsfgsdfgs",
               "title" : Blah blah blah,
               "authors" : ["Mr. Blah", "Dr. Blup"]},
             { "id" : "btbtbt.btbtbtbt",
               "title" : Ding ding ding ding,
               "authors" : ["Mr. Ding Dong", "Dr. Swim Swam"]}
            ]
        """
        start_time = time.time()

        self.recommender.open_db_connection()

        # === First check if we have a valid ID in which case only return that result
        terms = search_input.split(" ")
        if len(terms) == 1 :
            try :
                paper_id = terms[0].strip()
                data = self.recommender.get_data(paper_id)
                data = [{
                    "id" : paper_id,
                    "title" : data["title"],
                    "authors" : data["authors"]
                }]
                return json.dumps({'data':data, 'duration':0.0})
            except UnknownIDException :
                pass

        # === Next check if any terms contains a valid author last name
        author_result = self.search_per_author(search_input)
        
        # === Then check if any terms contains any title word
        title_result = self.search_per_title(search_input)

        # === If a term did match an author, search only papers by this author
        # and return results where the title matches any of the terms
        related_papers = set()
        if len(author_result) > 0 and len(title_result) > 0:
            related_papers = set.intersection(author_result, title_result)
        else:
            # Else return results for authors/title
            if len(author_result) > 0:
                related_papers = author_result
            if len(title_result) > 0:
                related_papers = title_result

        related_papers_data = []
        for paper_id in related_papers:
            data = self.recommender.get_data(paper_id)
            related_papers_data.append({
                "id" : paper_id,
                "title" : data["title"],
                "authors" : data["authors"].split("|")
                })

        # Compute search query duration in seconds
        duration = "{:.2f}".format(time.time() - start_time)

        return json.dumps({"data":related_papers_data, "duration":duration})

