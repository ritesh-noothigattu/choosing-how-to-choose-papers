import numpy as np
from sys import argv
import csv

# Order of features: "Quality_of_writing", "Originality", "Relevance", "Significance", "Technical_quality"

### Reading the aggregate function ###

aggr = {}
file = open('p1q1_aggregate_fn.txt', 'r')

freqs = [[0 for j in range(11)] for i in range(5)]

for line in file:
	# Each line is a function value and looks like "(6.0, 4.0, 7.0, 5.0, 4.0) : 4.000000000003894"
	split_line = line.split(':')
	x_val = split_line[0]
	y_val = float(split_line[1])

	x_val = x_val.replace('(', '')
	x_val = x_val.replace(')', '')
	x_val = tuple([int(float(v)) for v in x_val.split(',')])

	if(x_val in aggr):
		print("There should not be a duplicate entry in file!")
		exit(0)

	aggr[x_val] = y_val

	freqs[0][x_val[0]] += 1
	freqs[1][x_val[1]] += 1
	freqs[2][x_val[2]] += 1
	freqs[3][x_val[3]] += 1
	freqs[4][x_val[4]] += 1

for row in freqs:
	print(row)

### Done reading perfectly! ###

features = ['q', 'o', 'r', 's', 't']
full_features = ['Writing', 'Originality', 'Relevance', 'Significance', 'Technical']
rev_map = {'quality':0, 'originality':1, 'relevance':2, 'significance':3, 'technical':4}

feat1 = rev_map[argv[1]]
val1 = int(argv[2])
feat2 = rev_map[argv[3]]
val2 = int(argv[4])
feat3 = rev_map[argv[5]]
val3 = int(argv[6])

(free1, free2) = tuple(set(range(5)) - set([feat1, feat2, feat3]))	# The unset variables

empty_list = [None, None, None, None, None]
empty_list[feat1] = val1
empty_list[feat2] = val2
empty_list[feat3] = val3

x_axis = []
y_axis = []
z_axis = []

for i in range(1, 11):	# value for free1
	for j in range(1, 11):	# value for free2
		this_list = empty_list[:]
		this_list[free1] = i
		this_list[free2] = j
		this_tuple = tuple(this_list)

		if(this_tuple in aggr):
			x_axis.append(i)
			y_axis.append(j)
			z_axis.append(aggr[this_tuple])



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x_axis, y_axis, z_axis, linewidth=0.2, antialiased=True)

ax.set_xlabel(full_features[free1])
ax.set_ylabel(full_features[free2])
ax.set_zlabel('Aggregate overall')

# ax.set_xlim([1, 10])
# ax.set_ylim([1, 10])
# ax.set_zlim([3,6])

plt.savefig('aggregate_%s%d%s%d%s%d.pdf' % (features[feat1],val1,features[feat2],val2,features[feat3],val3), bbox_inches='tight', pad_inches = 0.1)
plt.close(fig)
