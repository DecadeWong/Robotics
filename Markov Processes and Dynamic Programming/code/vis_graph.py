""" 
A simple tool to visualize the graph input. 
Author: Yongxi Lu
"""

from graphviz import Digraph
import os.path as osp
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Visualize the weighted directed graph.')
	parser.add_argument('input_file', type=str, help='name of the input file.')
	parser.add_argument('output_file', type=str, default=None, help='name of the output file.')

	args = parser.parse_args()

	input_file = args.input_file
	output_file = args.output_file

	G = Digraph(filename=osp.splitext(osp.basename(args.input_file))[0], format='pdf', engine='fdp')
	G.attr('node', colorscheme='accent3', color='1', shape='oval', style="filled", label="")

	G_meta = [0, 0, 0]
	# read the graph from input file
	with open(input_file, 'r') as f:
		for i, line in enumerate(f):
			if i >= 3:
				v = line.split()
				color_id = min(max(1, int(float(v[2]))), 11)
				G.edge(v[0], v[1], colorscheme="rdylbu11", color="{:d}".format(color_id))
			else:
				G_meta[i] = int(line)

	# read the minimum cost path from the output file (if provided)
	path_nodes = []
	if output_file is not None:
		with open(output_file, 'r') as f:
			for i, line in enumerate(f):
				if i == 0:
					path_nodes = line.split()
	for n in path_nodes:
		G.node(n, colorscheme='accent3', color='3', shape='oval', style="filled")

	G.view()