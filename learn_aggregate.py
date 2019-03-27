import numpy as np
from sys import argv
import csv

p = int(argv[1])
q = int(argv[2])

allData = np.genfromtxt("Data/toy_data.csv", dtype = str, delimiter = ',')

header = allData[0,:]
allData = allData[1:,:]

paper_ind = np.where(header == "PaperID")[0][0]
reviewer_ind = np.where(header == "ReviewerID")[0][0]

quality_ind = np.where(header == "Quality_of_writing")[0][0]
originality_ind = np.where(header == "Originality")[0][0]
relevance_ind = np.where(header == "Relevance")[0][0]
signific_ind = np.where(header == "Significance")[0][0]
technical_ind = np.where(header == "Technical_quality")[0][0]

state_ind = np.where(header == "State")[0][0]

overall_ind = np.where(header == "Overall")[0][0]

paper_dict = {}	# Dictionary from paper_id to index in X/Y
rev_paper_dict = []	# Dictionary from index in X/Y to paper_id
m = 0	# number of papers

for paper_id in allData[:,paper_ind]:
	if paper_id in paper_dict:
		continue
	paper_dict[paper_id] = m
	rev_paper_dict.append(paper_id)
	m += 1

print("\n# of papers:", m)

reviewer_dict = {}	# Dictionary from reviewer_id to index in X/Y
n = 0	# number of reviewers

for rev_id in allData[:,reviewer_ind]:
	if rev_id in reviewer_dict:
		continue
	reviewer_dict[rev_id] = n
	n += 1

print("# of reviewers:", n)

d = 5	# number of features
if(np.shape(allData)[1] != (d+5)):
	print("Number of features not matching")
	exit(0)

feature_dict = {}	# Dictionary from feature vector to index in f
reverse_dict = []	# Dictionary from index in f to corresponding feature vector
size = 0

accepted_state = [-1 for j in range(m)]

X = (-1)*np.ones((n, m), dtype=int)	# X[i,j] gives index of feature score vector given by reviewer i to paper j
Y = (-float('inf'))*np.ones((n, m))	# Y[i,j] is the overall score by reviewer i to paper j

for row in allData:
	i = reviewer_dict[row[reviewer_ind]]
	j = paper_dict[row[paper_ind]]

	# Reading the x-values of this review
	this_scores = []
	this_scores.append(float(row[quality_ind]))
	this_scores.append(float(row[originality_ind]))
	this_scores.append(float(row[relevance_ind]))
	this_scores.append(float(row[signific_ind]))
	this_scores.append(float(row[technical_ind]))

	this_scores = tuple(this_scores)

	if this_scores in feature_dict:
		X[i,j] = feature_dict[this_scores]
	else:
		feature_dict[this_scores] = size
		reverse_dict.append(this_scores)
		size += 1
		X[i,j] = size-1

	Y[i,j] = float(row[overall_ind])

	state_value = (row[state_ind] == "Accepted")
	if(accepted_state[j] == -1):
		accepted_state[j] = state_value
	else:
		if(accepted_state[j] != state_value):
			print("Paper (" + rev_paper_dict[j] + ") has different decisions")
			exit(0)

num_accepted = 0
for j in range(m):
	if(accepted_state[j]):
		num_accepted += 1


import cvxpy as cvx
f = cvx.Variable(size)	# we have a variable for f-value on each of the x-values of the data

loss = 0
# Constructing the loss function
for i in range(n):

	this_y = []
	x_indices = []
	for j in range(m):
		if(Y[i,j] == -float('inf')):	# j was not reviewed by i
			continue
		this_y.append(Y[i,j])
		x_indices.append(X[i,j])
	this_y = np.array(this_y)

	loss += cvx.pnorm(this_y - f[x_indices], p)**q

obj = cvx.Minimize(loss)

constraints = []
# Adding all pairwise monotonicity constraints
for j1 in range(size):
	for j2 in range(j1+1, size):
		vec1 = np.array(reverse_dict[j1])
		vec2 = np.array(reverse_dict[j2])
		if(sum(vec1 >= vec2) == d):
			constraints.append(f[j1] >= f[j2])
		if(sum(vec2 >= vec1) == d):
			constraints.append(f[j2] >= f[j1])
print("\n# of (pairwise) constraints:", len(constraints))

prob = cvx.Problem(obj, constraints)
prob.solve()

print("\nOptimal objective:",prob.value)

print("\nAggregate function values:")
for j in range(size):
	print(reverse_dict[j], ':', f.value[j,0])

print()

paper_aggr_sc = [-1 for j in range(m)]
for j in range(m):
	this_scores = []
	for i in range(n):
		if(X[i,j] == -1):
			continue
		this_scores.append(f.value[X[i,j],0])

	if(q == 1):
		paper_aggr_sc[j] = np.median(this_scores)

	else:
		this_scores = np.array(this_scores)
		mu = cvx.Variable()
		loss = cvx.pnorm(this_scores - mu, q)
		obj = cvx.Minimize(loss)

		prob = cvx.Problem(obj)
		prob.solve()

		paper_aggr_sc[j] = mu.value

print("\nPaper aggregate scores:\n", paper_aggr_sc)

paper_aggr_sc = np.array(paper_aggr_sc)
paper_aggr_sc = -paper_aggr_sc	# Making values negative to get ranks in decreasing value of scores
order = np.argsort(paper_aggr_sc)
ranks = order.argsort()

our_accepted = set([])
for j in range(m):
	if(ranks[j] < num_accepted):
		our_accepted.add(j)
print("\nOur accepted:", our_accepted)

coincide_num = 0
for j in range(m):
	if(accepted_state[j]):	# j^th paper actually accepted
		if(j in our_accepted):
			coincide_num += 1

print("\nAccepted papers overlap:", str(coincide_num/num_accepted*100) + "%")
print()
