import json
import random

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
		'links' : map(lambda n : { 
			'target' : n['index'], 
			'source' : random.randint(0,100), 
			'value' : random.randint(0,100)
		}, nodes)
	}, indent=4)
