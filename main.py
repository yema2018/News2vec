import argparse
import numpy as np
import networkx as nx
import random_walks
from news2vec import newsfeature2vec

def parse_args():
	'''
	Parses the News2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run News2vec.")

	parser.add_argument('--input', nargs='?', default='graph/8500_cut.edgelist',
						help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/8500.emb',
						help='Embeddings path')

	parser.add_argument('--map', nargs='?', default='map/8500.map',
						help='Map indice to nodes')

	parser.add_argument('--include', nargs='?', default=True,
						help='Boolean including keyword nodes')

	parser.add_argument('--dimensions', type=int, default=128,
						help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=100,
						help='Length of walk per source. Default is 100.')

	parser.add_argument('--num-walks', type=int, default=5,
						help='Number of walks per source. Default is 5.')

	parser.add_argument('--window-size', type=int, default=10,
						help='Context size for optimization. Default is 10.')

	parser.add_argument('--p', type=float, default=1,
						help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
						help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=True)

	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''

	map_dict = {}
	with open(args.map, 'r',encoding='utf-8') as f:
		for l in f:
			l = l.strip('\n').split(' ')
			map_dict[l[0]] = l[1]

	walks1 = list()
	for walk in walks:
		walks1 += list(map(lambda x: map_dict[str(x)], walk))
	print(walks1[:100])
	newsfeature2vec(walks1,args.output,include=args.include,skip_window=args.window_size,iter=int(len(walks1)))
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph()
	G = random_walks.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	print(len(walks),walks[0])
	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
