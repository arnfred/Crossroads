import json
import random

import recommender
reload(recommender)
from recommender import Recommender

# Init recommender
recommender = Recommender()
# Load feature vectors (may be long)
recommender.load_feature_vectors('data/feature_vectors_2013_K200.cpkl')
# Load id to title map
recommender.load_id_to_title_map('data/id_to_title_map.db')
# Build tree (may be long)
recommender.build_tree(metric='euclidean')

def center(paper_id, k = 100) :
	""" return a graph with the k nearest neighbors of the paper_id """
	distances, indices = recommender.get_nearest_neighbors(paper_id, k)
	nodes = [{ 'index' : neighbor_id, 'title': recommender.id_to_title_map[neighbor_id]} \
			for neighbor_id in recommender.ids[indices]]
	links = [{'source' : paper_id, 'target' : neighbor_id, 'weight': dist} \
			for neighbor_id, dist in zip(recommender.ids[indices], distances)]
	return json.dumps({'nodes' : nodes, 'links' : links})


def getTEMPgraph(n) :

	nodes  = [{ 'index' : i } for i in range(n)]
	return json.dumps({
		'nodes' : nodes,
		'links' : map(lambda i : { 'target' : i, 'source' : random.randint(0,99), 'value' : random.randint(0,99)}, range(n))
	}, indent=4)
