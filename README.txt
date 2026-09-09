

BBC News Text Mining: Clustering & Topic Modeling

This notebook implements an NLP pipeline for the unsupervised analysis of the BBC News dataset, focusing on Clustering and Topic Modeling. By benchmarking these algorithms, we aim to justify the most effective model for short-text analysis and provide actionable insights into news categorization, trend detection, and archive organization

The code performs the following operations:

1. Preprocessing: Text cleaning, punctuation, and stopword removal (NLTK).

2. Vectorization: Conversion of text into numerical vectors.

3. Clustering: Article segmentation using K-Means.

4. Topic Modeling: Extraction of latent themes using Latent Dirichlet Allocation (LDA).

5. Evaluation: Calculation of quality metrics (Silhouette Score, Davies-Bouldin, Perplexity).


The notebook handles dependency installation. The main libraries are:

- nltk, scikit-learn, pandas, numpy

- matplotlib, seaborn, wordcloud

Usage

The project is optimized for Google Colab:

Upload the .ipynb file to Colab.

Run all cells (Runtime > Run all).

Upon completion, the notebook will automatically show all the results.