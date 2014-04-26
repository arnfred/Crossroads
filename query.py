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
    graph = Graph(recommender.get_title)
    print(graph)
    print(k)
    # Populate graph
    if paper_id != "":
        graph_walk(graph, [(paper_id, 0)], k = k, visited = {})
    # Create json
    graph_json = graph.to_JSON()
    return graph_json


def graph_walk(graph, to_visit = [], k = 11, max_level = 4, visited = {}) :
    """ Given a paper id, we find the nearest neighbors and add them to the
    graph, then find the nearest neighbors of these and add those to the graph
    recursing k steps down """
    # Let's explore the graph if there is anything left to explore
    while len(to_visit) > 0 :
        parent_id, level = to_visit.pop()
        visited[parent_id] = True
        distances, indices = recommender.get_nearest_neighbors(parent_id, k)
        print(distances, indices)
        for dist, node_id in zip(distances, recommender.ids[indices]) :
            # Anyway, add node to graph (if it exists, the new link is added)
            graph.add(node_id, parent_id, dist)
            # If we haven't seen this node we continue
            if node_id not in visited and level + 1 <= max_level :
                to_visit.append((node_id, level + 1))


class Graph :
    """ Helper class for constructing a graph """

    def __init__(self, get_title) :
        self.nodes = {}
        self.get_title = get_title


    def add(self, node_id, parent_id, distance) :
        """ Add one node to the graph """
        # Checks if node exists already
        if node_id not in self.nodes :
            self.nodes[node_id] = {
                'title' : self.get_title(node_id),
                'links' : {
                    parent_id : int(100*distance)
                }
            }

        # If not, then add the link to the parent to the map of links
        else :
            self.nodes[node_id]['links'][parent_id] = distance


    def exists(self, node_id) :
        """ Function to check if node exists in graph already """
        return node_id in self.nodes


    def to_JSON(self) :
        """ Converts the graph to json """
        # Create map of nodes
        nodes = [{ 'id' : node_id, 'title' : val['title'] }
                 for node_id, val in self.nodes.iteritems()]
        # Create map of indices
        idx = { val['id']:i for i, val in enumerate(nodes) }
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
