import json
from itertools import chain
import numpy as np


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
	global graph
	graph = Graph(recommender.get_data)
	# Populate graph
	if paper_id != "":
		distances, indices, methods_dist, methods_idx = graph_walk(recommender, graph, to_visit = [(paper_id,0)], k = k, max_level = 2, visited = {})
		# distances, indices, methods_dist, methods_idx = graph_walk_v2(recommender, graph, paper_id, k)
	graph_dict = graph.to_dict()

	main_node_data = {'final_recommendation':[], 'LDABasedRecommendation':[], 'AuthorBasedRecommendation':[]}
	for rank,(idx,dist) in enumerate(zip(indices[1:k],distances[1:k])):
		doc_id = recommender.ids[idx]
		element = {}
		element['rank'] = rank+1
		element['total_distance'] = "%.4f"%(1-abs(dist))
		element['id'] = doc_id
		element['topics_distance'] = "%.4f"%(1-methods_dist['LDABasedRecommendation'][idx])
		element['authors_distance'] = "%.4f"%(1-methods_dist['AuthorBasedRecommendation'][idx])
		main_node_data['final_recommendation'].append(element)

	for name,idx in methods_idx.iteritems():
		indices_sorted = np.argsort(methods_dist[name])[1:k]
		distances_sorted = np.sort(methods_dist[name])[1:k]
		for rank,(idx,dist) in enumerate(zip(indices_sorted,distances_sorted)):
			doc_id = recommender.ids[idx]
			element = {}
			element['rank'] = rank+1
			element['total_distance'] = "%.4f"%(1-abs(dist))
			element['id'] = doc_id
			element['topics_distance'] = "%.4f"%(1-methods_dist['LDABasedRecommendation'][idx])
			element['authors_distance'] = "%.4f"%(1-methods_dist['AuthorBasedRecommendation'][idx])
			main_node_data[name].append(element)

	return json.dumps({'graph_data':graph_dict, 'main_node_data':main_node_data})


def graph_walk(recommender, graph, to_visit = [], k = 5, max_level = 2, visited = {}) :
	""" Given a paper id, we find the nearest neighbors and add them to the
	graph, then find the nearest neighbors of these and add those to the graph
	recursing k steps down """
	# Let's explore the graph if there is anything left to explore
	while len(to_visit) > 0 :
		parent_id, level = to_visit.pop()
		visited[parent_id] = True
		graph.add(parent_id, parent_id, 0.0, 1)  
		distances, indices, methods_dist, methods_idx = recommender.get_nearest_neighbors(parent_id, k)  
		if len(visited) == 1:
			center_node_data = distances, indices, methods_dist, methods_idx
		for dist, node_id in zip(distances, recommender.ids[indices]) :
			# Anyway, add node to graph (if it exists, the new link is added)
			graph.add(node_id, parent_id, dist, level + 1)
			# If we haven't seen this node we continue
			if node_id not in visited and level < max_level-1 :
				to_visit.append((node_id, level + 1))
	return center_node_data

def graph_walk_v2(recommender, graph,  paper_id, k = 5) :
	""" Given a paper id, we find the nearest neighbors and add them to the
	graph, then find the nearest neighbors of these and add those to the graph
	recursing k steps down """
	
	parent_id = paper_id
	graph.add(paper_id, parent_id, 0.0, 1)  
	distances, indices, methods_dist, methods_idx = recommender.get_nearest_neighbors_online(parent_id, k)  
	center_node_data = distances, indices, methods_dist, methods_idx
	
	for dist, node_id in zip(distances, recommender.ids[indices]) :
		# Anyway, add node to graph (if it exists, the new link is added)
		graph.add(node_id, parent_id, dist, 1)
	
	for parent_id in recommender.ids[indices]:
		distances, indices, methods_dist, methods_idx = recommender.get_nearest_neighbors_online(node_id, k)  
		for dist, node_id in zip(distances, recommender.ids[indices]) :
			if parent_id in graph.nodes and node_id in graph.nodes:
				graph.nodes[node_id]['links'][parent_id] = dist

	return center_node_data

class Graph(object) :
	""" Helper class for constructing a graph """

	def __init__(self, get_data) :
		self.nodes = {}
		self.get_data = get_data

	def add(self, node_id, parent_id, distance, level) :
		""" Add one node to the graph """
		data = self.get_data(node_id)
		abstract = data['abstract']
		title = data['title']
		authors = data['authors'].replace('|', ', ')+u'\n'

		node_id = node_id
		parent_id = parent_id

		# If this is a link between the same node, then just add the node without links
		if node_id == parent_id and node_id not in self.nodes :
			self.nodes[node_id] = {
				'title'   : title,
				'level'   : level - 1,
				'abstract': abstract,
				'authors' : authors,
				'parent_id' : parent_id,
				'links'   : {}
			}

		# Checks if node exists already
		if node_id not in self.nodes :
			self.nodes[node_id] = {
				'title' : title,
				'level' : level,
				'abstract': abstract,
				'authors' : authors,
				'parent_id' : parent_id,
				'links' : {
					parent_id : distance
				}
			}

		# If not, then add the link to the parent to the map of links
		else :
			self.nodes[node_id]['links'][parent_id] = distance


	def exists(self, node_id) :
		""" Function to check if node exists in graph already """
		return node_id in self.nodes


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
				 for node_id, val in self.nodes.iteritems()]
		# Create map of indices
		idx = { val['id']:i for i, val in enumerate(nodes) }
		# Create list of links
		links = self.flatten([self.make_links(n, nid, idx)
							  for nid, n in self.nodes.iteritems()])
		return {'nodes' : nodes, 'links' : links}


	def make_links(self, node, node_id, idx) :
		""" Produces a list of correctly formatted links given a node, an id
			and an index map """
		return [{ 'source' : idx[node_id], 'target' : idx[key], 'value' : dist }
				for key, dist in node['links'].iteritems() ]


	def flatten(self, list_list) :
		return list(chain.from_iterable(list_list))

