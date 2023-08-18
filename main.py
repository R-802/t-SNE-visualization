import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(42)
n_samples = 300
n_features = 50
X = np.random.randn(n_samples, n_features)

# Perform t-SNE embedding
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)

# Visualize the embedded data
plt.figure(figsize=(10, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='b', marker='o')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
