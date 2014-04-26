import json
import recommender
reload(recommender)
from recommender import Recommender
from itertools import chain

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
    g = Graph(recommender.get_title)
    # Populate graph
    if paper_id != "":
        distances, indices = recommender.get_nearest_neighbors(paper_id, k)
        for dist, node_id in zip(distances, recommender.ids[indices]) :
            g.add(node_id, paper_id, dist)
    # Create json
    graph_json = g.to_JSON()
    print(graph_json)
    return graph_json



class Graph :
    """ Helper class for constructing a graph """

    def __init__(self, get_title) :
        self.nodes = {}
        self.get_title = get_title


    def add(self, id, parent_id, distance) :
        """ Add one node to the graph """
        # Checks if node exists already
        if not self.nodes.get(id, False) :
            self.nodes[id] = {
                'title' : self.get_title(id),
                'links' : {
                    parent_id : int(100*distance)
                }
            }
        # If not, then add the link to the parent to the map of links
        else :
            self.nodes[id]['links']['parent_id'] = distance


    def to_JSON(self) :
        """ Converts the graph to json """
        # Create map of nodes
        nodes = [{ 'id' : node_id, 'title' : val['title'] }
                 for node_id, val in self.nodes.iteritems()]
        # Create map of indices
        idx = { val['id']:i for i, val in enumerate(nodes) }
        print(idx)
        # Create list of links
        links = self.flatten([self.make_links(n, nid, idx)
                              for nid, n in self.nodes.iteritems()])
        return json.dumps({'nodes' : nodes, 'links' : links})


    def make_links(self, node, node_id, idx) :
        """ Produces a list of correctly formatted links given a node, an id
            and an index map """
        return [{ 'source' : idx[node_id], 'target' : idx[key], 'value' : dist }
                for key, dist in node['links'].iteritems() ]


    def flatten(self, list_list) :
        return list(chain.from_iterable(list_list))
