import json
import random

import recommender
reload(recommender)
from recommender import Recommender

def center(id, n = 100) :
	""" return a graph with the n nearest neighbors of the id """

	# Note: when the page is loaded, the function is called with id = ""

	# TODO ...

	# For now return random graph
	return getTEMPgraph(n)


def getTEMPgraph(n) :

	nodes  = [{ 'index' : i } for i in range(n)]
	return json.dumps({
		'nodes' : nodes,
		'links' : map(lambda i : { 'target' : i, 'source' : random.randint(0,99), 'value' : random.randint(0,99)}, range(n))
	}, indent=4)
