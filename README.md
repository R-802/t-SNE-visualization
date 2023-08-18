# t-SNE Visualization
## t-SNE (t-distributed Stochastic Neighbor Embedding)

t-SNE is a dimensionality reduction technique commonly used for visualizing high-dimensional data in lower dimensions while preserving pairwise similarities between data points. It is particularly effective at capturing complex relationships in the data that might be lost with other techniques.

### Algorithm Overview

t-SNE constructs a probability distribution over pairs of high-dimensional data points that represents their similarities. It then constructs a similar probability distribution over the corresponding points in a lower-dimensional space. The goal is to minimize the divergence between these two distributions. The algorithm uses a student-t distribution (t-distribution) to model the similarity between points.

### Objective Function

The t-SNE algorithm minimizes the Kullback-Leibler (KL) divergence between two probability distributions: \(P\) for pairwise similarities in the original high-dimensional space, and \(Q\) for pairwise similarities in the lower-dimensional space.

The objective function is defined as follows:

$$ C = KL(P || Q) = \sum_{ij} P_{ij} \log \frac{P_{ij}}{Q_{ij}}$$

- $P_{ij}\$: The pairwise similarity between data points $i$ and $j$ in the original high-dimensional space. This is typically computed using Gaussian kernel or normalized distances.

- $Q_{ij}\$: The pairwise similarity between data points $i$ and $j$ in the lower-dimensional space. This is typically computed using a student-t distribution.

- $C\$: The cost function, which measures the Kullback-Leibler divergence between $P_{ij}$ and $Q_{ij}$. The optimization process aims to minimize this divergence.

The optimization process adjusts the positions of the points in the lower-dimensional space in such a way that the divergence between $P_{ij}$ and $Q_{ij}$ is minimized. This helps in creating an embedding that maintains the pairwise similarities as much as possible.

### Perplexity

Perplexity is a hyperparameter in t-SNE that controls the balance between focusing on local and global aspects of the data. It is used to determine the variance of the Gaussian distribution used to compute conditional probabilities in the high-dimensional space. A higher perplexity value encourages the algorithm to consider more points as neighbors, resulting in a smoother embedding.

### Steps of the Algorithm

1. Compute pairwise Euclidean distances between data points in the high-dimensional space.
2. Convert the Euclidean distances to conditional probabilities using a Gaussian distribution with a specified perplexity value.
3. Initialize the embeddings randomly in the lower-dimensional space.
4. Compute pairwise Euclidean distances between embedded points in the lower-dimensional space.
5. Convert the Euclidean distances to conditional probabilities using a student-t distribution.
6. Minimize the KL divergence between the two distributions by adjusting the positions of points in the lower-dimensional space using gradient descent.

### Usage

Simply provide your high-dimensional data and specify hyperparameters like perplexity and learning rate to obtain a lower-dimensional embedding suitable for visualization.

For example:
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_embedded = tsne.fit_transform(X)
```

### Conclusion
t-SNE is a powerful technique for visualizing high-dimensional data in a lower-dimensional space, allowing the exploration of complex relationships in the data. It is particularly useful for identifying clusters and patterns that might not be easily visible in the original space.
