import json
from itertools import chain
import numpy as np

import networkx as nx

def center(recommender, paper_id, k) :
	"""
	return JSON dump with a graph with the k nearest neighbors of the paper_id
	and the detail of the algorithm data

	Arguments:
		paper_id : string
			arXiv id of a paper in the dataset
			e.g. try with '1402.1774', '1304.5220', '1401.3127', '1307.7223', '1304.6026'
		k : int
			Number of neighbors
	"""
	recommender.open_db_connection()
	# Populate graph
	if paper_id != "":
		global graph	
		graph = myGraph(recommender.get_data)
		similarity, indices, methods_sim, methods_idx = graph_creation(recommender, paper_id, k)
	graph_dict = graph.to_dict()

	# Format the detail of the algorithm data and store it as the 'neighbors_data' key
	neighbors_data = {
		'LDABasedRecommendation': {'key':'Topic similarity', 'values':[]},
		'AuthorBasedRecommendation': {'key':'Author similarity', 'values':[]}
	}

	for rank,(idx,dist) in enumerate(zip(indices[1:],similarity[1:])):
		doc_id = recommender.ids[idx]
		e1 = {}
		e1['x'] = rank
		e1['xlabel'] = doc_id
		e1['y'] = methods_sim['LDABasedRecommendation'][idx]
		neighbors_data['LDABasedRecommendation']['values'].append(e1)

		e2 = {}
		e2['x'] = rank
		e2['xlabel'] = doc_id
		e2['y'] = methods_sim['AuthorBasedRecommendation'][idx]
		neighbors_data['AuthorBasedRecommendation']['values'].append(e2)

	return json.dumps({'graph_data':graph_dict, 'neighbors_data':neighbors_data.values()})


def graph_creation(recommender, paper_id, k):
	
	level = 0
	max_level = 2
	to_visit = [(paper_id, level)]
	visited = set()
	parent_id = paper_id # Set the parent of the root node as itslef

	# Init root node
	graph.add(paper_id, parent_id, level, 0.0)

	while len(to_visit) > 0:
		parent_id, level = to_visit.pop()
		new_level = level+1

		distances, indices, methods_sim, methods_idx = recommender.get_nearest_neighbors(parent_id, k)
		visited.add(parent_id)

		# Keep track of the neighbors of the root node
		if parent_id is paper_id:
			center_node_data = distances, indices, methods_sim, methods_idx

		for dist, idx in zip(distances, indices):
			child_id = recommender.ids[idx]
			if new_level <= max_level:
				graph.add(child_id, parent_id, new_level, dist)
				if child_id not in visited:
					to_visit.append((child_id, new_level))

	# Remove nodes at max_level that have only one parent
	for node,data in graph.nodes(data=True):
		if data['level'] == max_level and graph.in_degree(node) == 1:
			graph.remove_node(node)

	return center_node_data


class myGraph(nx.DiGraph) :
	""" Helper class for constructing a graph """

	def __init__(self, get_data) :
		super(myGraph, self).__init__()
		self.get_data = get_data

	def add(self, node_id, parent_id, level, distance) :
		""" Add one node to the graph with its data"""
		# Get article data
		data = self.get_data(node_id)
		
		# Add node to the graph
		if not node_id in self.nodes():
			super(myGraph, self).add_node(node_id,
				abstract=data['abstract'], 
				title=data['title'],
				authors=data['authors'].replace('|', ', ')+u'\n',
				level=level,
				parent_id=parent_id)

		# Add a link from the parent to the child
		self.add_edge(parent_id, node_id, distance=distance)

	def to_dict(self) :
		""" Converts the graph to dict """
		# Create map of nodes
		nodes = [{
			'id' : node_id,
			'title' : val['title'],
			'abstract' : val['abstract'],
			'authors' : val['authors'],
			'level' : val['level'],
			'parent_id' : val['parent_id']
			}
				for node_id, val in self.nodes(data=True)]
		# Create map of indices
		idx = { val['id']:i for i, val in enumerate(nodes) }
		# Create list of links
		links = [{ 
			'source' : idx[source], 
			'target' : idx[target], 
			'value' : val['distance'] 
			}
				for source,target,val in self.edges(data=True)]
		return {'nodes' : nodes, 'links' : links}

	def flatten(self, list_list) :
		return list(chain.from_iterable(list_list))

