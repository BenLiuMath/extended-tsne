# extended-tsne

This repository contains the report and poster for a project for the class CSE5 546: Machine Learning. I worked on this project with Megan Morrison, another student from Applied Math at the University of Washington.

Our work was aimed at modifying and potentially improving the t-SNE algorithm based on our understanding of the intent of the algorithm.

## Summary of SNE and t-SNE
t-SNE stands for t-distributed Stochastic Neighbor Embedding (SNE), and is a specific type or extension of SNE. The aim of SNE is to visually cluster high-dimensional data, by projecting the data from a high-dimensional origin space, onto a low-dimensional embedding space. The embedding space can be two or three dimensions for easy viewing in 2D or 3D plots.

SNE aims to preserve or meaningfully represent distances in the original high-dimensional data. Roughly speaking, a distribution on the relative distances between data points is calculated in the origin space. Closer points receive a higher weight or probability; the magnitudes are essentially governed by a normal distribution, with more distant points rapidly dropping off in relative probability or weight.

A similar distribution is calculated for points in the embedding space, and it is the goal of SNE to bring these distributions into close agreement. Close agreement between the distributions means that distances, and therefore some quality of 'clustered points', is faithfully reproduced in the visualization. Distances cannot be exactly reproduced in the embedding space, so this is achieved through optimization. The procedure begins with points randomly initialized in the embedding space, which earns the 'stochastic' in SNE. Optimization is performed with gradient descent with the Kullback-Leibler divergence serving as the objective function by measuring the 'difference' between the two distributions.

t-SNE is the insight to use Student's t-distributions rather than normal distributions in the calculation of the probability distribution or relative weights in the embedding space. The t-distribution family has fat tails, or more probability mass assigned to greater values. This allows the correspondence between distances in the high- and low-dimensional spaces to be adjusted to account for the curse of dimensionality, where distances in high-dimensional data are intrinsically greater.

## Modifying t-SNE
My main insight in this project was based on my knowledge of the t-distribution. The t-distribution is a parametrized family of distributions; the parameter is often denoted 'nu' and referred to as the degrees of freedom. This distribution is commonly used in statistics. The t-distribution family has the property that it converges to the normal distribution as nu is taken to infinity. My thought was that instead of using a normal distribution to calculate weights in an origin space of dimension N, we could use a t-distribution with nu=N. This change has a number of consequences

1. The modification has very little effect when the dimension N is large; a t-distribution with large degree of freedom is very close to the normal distribution. The algorithm would therefore behave very similar to normal t-SNE under typical use-cases.
2. t-SNE is not typically used on low-dimensional data, but it is possible. Mathematically, it is strange to think that applying the data to low-dimensional data should result in much change; for example, running the algorithm on two-dimensional data and embedding into a two-dimensional space should in principle simply return a visualization of the original data, since distances are already perfectly represented. The proposed change might make this possible.


## Results
Our modifications had mixed results. The modified algorithm was not in general superior.

The goal of t-SNE is usually to view the finished product only, but in our implementations of the algorithm we animated and viewed the progress of the optimization. We found that the modified algorithm produced interesting transient behaviors. At times during the optimization, the algorithm produced very clear clusters that were more distinguished than that of classical t-SNE, but these transient clusters often disappeared as the optimization procedure went on. This indicates an interesting tendency of the system to generate troughs or shadow attractors. In these cases, the gradient descent algorithm fell into the trough where progress towards the minimum was slow. In this trough, the clear clusters were apparent. As the optimization continued and fell out of the trough into a deeper local minimum, the clear clusters were lost.


## Other ideas
I also had the idea that the origin and embedding distributions could both be t-distributions, and that rather than embedding high-dimensional data directly, we might be able to step down from high to low dimension gradually. This would allow us to remove the stochastic element of the algorithm, by using the initial data itself as the inital condition. This ultimately was not very successful.