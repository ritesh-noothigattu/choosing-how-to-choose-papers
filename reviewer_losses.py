import numpy as np
from sys import argv
import csv

# p & q taken to be 1 for this file

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

reviewer_dict = {}	# Dictionary from reviewer_id to index in X/Y
n = 0	# number of reviewers

for rev_id in allData[:,reviewer_ind]:
	if rev_id in reviewer_dict:
		continue
	reviewer_dict[rev_id] = n
	n += 1

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


############# Reading the aggregate function

aggr = {}
file = open('p1q1_aggregate_fn.txt', 'r')

for line in file:
	# Each line is a function value and looks like "(6.0, 4.0, 7.0, 5.0, 4.0) : 4.000000000003894"
	split_line = line.split(':')
	x_val = split_line[0]
	y_val = float(split_line[1])

	x_val = x_val.replace('(', '')
	x_val = x_val.replace(')', '')
	x_val = tuple([float(v) for v in x_val.split(',')])

	if(x_val in aggr):
		print("There should be a duplicate entry in file!")
		exit(0)

	aggr[x_val] = y_val


############# Done reading perfectly!

rev_losses = [0 for i in range(n)]

for i in range(n):	# computing the loss for each reviewer
	this_losses = []
	for j in range(m):
		if(Y[i,j] == -float('inf')):	# j was not reviewed by i
			continue
		this_losses.append(abs(Y[i,j] - aggr[reverse_dict[X[i,j]]]))
	this_losses = np.array(this_losses)
	rev_losses[i] = np.mean(this_losses)	# Taking a mean to normalize scale across reviewers #np.sum(this_losses)	# Taking simple sum since p = 1

rev_losses = np.array(rev_losses)


import matplotlib.pyplot as plt

print(plt.hist(rev_losses, bins = np.arange(0,9.2,step=0.20), edgecolor='black'))
plt.xlabel('Normalized loss of reviewer',fontsize=30, labelpad = 10)
plt.ylabel('Frequency',fontsize=30, labelpad = 10)

plt.xticks(range(10))

plt.tick_params(axis='x',labelsize=20)
plt.tick_params(axis='y',labelsize=20)

fig = plt.gcf()

plt.show()

print('\nTotal loss:', np.sum(rev_losses))
print('Average loss per reviewer:', np.mean(rev_losses))
print('Standard deviation:', np.std(rev_losses))
