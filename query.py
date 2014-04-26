import json
import random
import recommender
reload(recommender)
from recommender import Recommender

# Init recommender
def init_recommender() :
    # Init recommender
    recommender = Recommender()
    # Load feature vectors (may be long)
    recommender.load_feature_vectors('data/feature_vectors_2013_K200.cpkl')
    # # Load id to title map
    # Build tree (may be long)
    recommender.build_tree(metric='euclidean')

    return recommender
recommender = init_recommender()

# WARNING : Note that importing this module might take a up to a minute in order
# to load the recommender data. But this is only done once during the import.

def center(paper_id, k) :
    """
    return a graph with the k nearest neighbors of the paper_id

    Arguments:
        paper_id : string
            arXiv id of a paper in the dataset
            e.g. try with '1402.1774', '1304.5220', '1401.3127', '1307.7223', '1304.6026'
        k : int
            Number of neighbors

    Returns:
        json graph with the following description:
            - 'nodes' are composed of:
                - an 'index' : the arXiv id
                - a 'title' : the title of the paper
            - 'links' are composed of:
                - a 'source' : the id of the paper if are querying (i.e. paper_id)
                - a 'target' : the id of the neighbor
                - a 'value' : the distance between the paper and its neighbor
                    (i.e. the smaller it is, the closer the papers are)
    """
    # recommender.load_id_to_title_map('data/id_to_title_map.db')
    recommender.open_db_connection('data/arxiv.db')
    if paper_id != "":
        distances, indices = recommender.get_nearest_neighbors(paper_id, k)
        nodes = [{ 'id' : neighbor_id, 'title': recommender.get_title(neighbor_id)} \
                 for i, neighbor_id in zip(indices, recommender.ids[indices])]
        links = [{'source' : i, 'target' : 0, 'value': int(100*dist)} \
                 for i, dist in enumerate(distances)]
    else:
        nodes = []
        links = []
    return json.dumps({'nodes' : nodes, 'links' : links})


def getTEMPgraph(n) :

    nodes  = [{ 'title' : "Yup" } for i in range(n)]
    return json.dumps({
        'nodes' : nodes,
        'links' : map(lambda i : { 'target' : i+4, 'source' : random.randint(0,99),
                                  r'value' : random.randint(0,99)}, range(n))
    }, indent=4)
