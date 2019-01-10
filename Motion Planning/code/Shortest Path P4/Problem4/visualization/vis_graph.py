""" 
A simple tool to visualize a graph. 
Author: Yongxi Lu

Example usage:
python graph_vis.py input.txt coords.txt –output_file=output.txt
"""

from graphviz import Digraph
import os.path as osp
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Visualize the weighted directed graph.')
	parser.add_argument('input_file', type=str, help='name of the input file.')
	parser.add_argument('coords_file', type=str, help='name of the coordinates file')
	parser.add_argument('--output_file', type=str, default=None, help='name of the output file.')

	args = parser.parse_args()

	input_file = args.input_file
	coors_file = args.coords_file
	output_file = args.output_file

	G = Digraph(filename=osp.splitext(osp.basename(args.input_file))[0], format='pdf', engine='neato')
	G.attr('node', colorscheme='accent3', color='1', shape='circle', style="filled", label="", pin="True")

	# assign nodes to coordinates
	with open(args.coords_file, 'r') as f:
		for i, line in enumerate(f):
			v = line.split()
			v = [float(v[0]), float(v[1])]
			G.node(str(i+1), pos="{:f},{:f}".format(2 * v[1], -2 * v[0]))

	G_meta = [0, 0, 0]

	# read edges to figure out color scaling
	weights = []
	with open(input_file, 'r') as f:
		for i, line in enumerate(f):
			if i >= 3:
				v = line.split()
				weights.append(float(v[2]))
	max_w = max(weights)
	min_w = min(weights)
	diff_w = max_w - min_w + 1e-5

	# read the graph from input file
	with open(input_file, 'r') as f:
		for i, line in enumerate(f):
			if i >= 3:
				v = line.split()
				w = float(v[2])
				color_id = min(max(int(((w - min_w) / diff_w * 10) + 1), 1), 11)
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
		G.node(n, colorscheme='accent3', color='3', shape='circle', style="filled")

	G.view()
