# choosing-how-to-choose-papers

Code corresponding to the paper [Choosing How to Choose Papers](http://www.cs.cmu.edu/~rnoothig/papers/subjective_reviews.pdf)

- `Data/toy_data.csv`: Toy data with completely random data, used to only depict the format of the input

- `learn_aggregate.py`: Main learning code that takes p and q as input, and performs L(p,q) aggregation on the input dataset to compute the aggregate function. Prints out relevant statistics. Save the printed "Aggregate function values" into say "p1q1_aggregate_fn.txt" for use by other files

- `p1q1_aggregate_fn.txt`: The aggregate function learned by applying L(1,1) aggregation to the toy data "Data/toy_data.csv"

- `plot3d_aggregate.py`: File that reads the aggregate function in "p1q1_aggregate_fn.txt" and plots it in 3-D fixing 3 features, varying the other 2. Takes as input the three features to be fixed, as well as values to set them to. (The toy aggregate function has too few values to be plotted in 3D for general values. An example that works is fixing quality as 3, relevance as 6, and significance as 5)

- `reviewer_losses.py`: File that reads the input dataset and the aggregate function in "p1q1_aggregate_fn.txt" to compute losses of reviewers. It plots the histogram of these losses, and prints some relevant statistics

- 'varying_num_revs.py': Similar to "learn_aggregate.py", except that this file takes p, q *and* a cap on the number of reviews per paper. It (randomly) discards reviews exceeding this cap, and performs L(p,q) aggregation on the remaining data