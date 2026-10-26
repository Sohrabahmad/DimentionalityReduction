# Principal Component Analysis (PCA) on the Iris Dataset

## 📌 Project Overview

This project demonstrates **Principal Component Analysis (PCA)**, a dimensionality reduction technique used to transform high-dimensional data into a smaller number of features while retaining as much variance as possible.

The project uses the Iris dataset from Scikit-learn, standardizes its four numerical features, and applies PCA to reduce them to two principal components. A scatter plot visualizes the transformed data using the original species labels.

## 🎯 Objectives

- Load and explore the Iris dataset.
- Standardize the input features using `StandardScaler`.
- Apply PCA to reduce four features to two dimensions.
- Calculate the explained variance ratio.
- Inspect the principal component directions.
- Visualize the reduced dataset in two dimensions.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data handling
- **Matplotlib** — Data visualization
- **Scikit-learn** — Dataset loading, feature scaling, and PCA
- **Jupyter Notebook** — Interactive development environment

## 📂 Project Structure

```text
PCA-Project/
│
├── PCA.ipynb
└── README.md
```

## 📊 Dataset Description

The project uses Scikit-learn's built-in Iris dataset. No external CSV file is required.

The dataset contains 150 flower samples, four numerical features, and three Iris species.

| Feature | Description |
|---|---|
| Sepal length | Length of the sepal |
| Sepal width | Width of the sepal |
| Petal length | Length of the petal |
| Petal width | Width of the petal |

The notebook stores the features in a Pandas DataFrame `X` and the original species labels in `y`.

## ⚙️ Project Workflow

### 1. Import Libraries and Load the Dataset

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()

X = pd.DataFrame(iris.data)
y = iris.target
```

### 2. Standardize the Features

PCA is sensitive to differences in feature scales. The notebook standardizes the four input features before applying dimensionality reduction.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

The scaled dataset has a shape of `(150, 4)`, representing 150 samples and four features.

### 3. Apply Principal Component Analysis

The notebook uses PCA to reduce the data from four dimensions to two.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

x_pca = pca.fit_transform(X_scaled)
```

The transformed dataset has a shape of `(150, 2)`.

### 4. Calculate Explained Variance

The explained variance ratio measures the proportion of the dataset's variance captured by each principal component.

```python
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

The notebook reports:

- **PC1:** 72.96%
- **PC2:** 22.85%
- **Combined explained variance:** 95.81%

Together, the two principal components retain approximately 95.81% of the variance in the standardized Iris dataset.

### 5. Inspect Principal Components

The notebook prints the component directions using:

```python
print(pca.components_)
```

These values describe how the original standardized features contribute to each principal component.

### 6. Visualize the Reduced Dataset

A scatter plot displays the two principal components, with colors determined by the original Iris species labels.

```python
plt.scatter(
    x_pca[:, 0],
    x_pca[:, 1],
    c=y
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA for iris dataset")

plt.show()
```

This visualization helps explore how the flower samples are distributed in the reduced two-dimensional space.

## 📈 Results and Observations

The notebook demonstrates the following results:

| Measurement | Result |
|---|---:|
| Original number of features | 4 |
| Reduced number of components | 2 |
| Number of samples | 150 |
| Variance explained by PC1 | 72.96% |
| Variance explained by PC2 | 22.85% |
| Total variance retained | 95.81% |

**Key finding:** PCA reduces the number of features by 50% while retaining approximately 95.81% of the variance in the standardized dataset.

The visualization provides a way to inspect the separation and overlap of the original Iris species in the reduced feature space.

## 🔍 Key Concepts

| Concept | Description |
|---|---|
| PCA | A dimensionality reduction technique that transforms features into principal components. |
| StandardScaler | Standardizes features to approximately zero mean and unit variance. |
| Principal Components | New, mutually orthogonal directions ordered by explained variance. |
| Explained Variance Ratio | Indicates how much variance each principal component captures. |
| Dimensionality Reduction | Reduces the number of features while preserving useful information. |

## 💻 Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd PCA-Project
```

Replace `<your-repository-url>` with your actual GitHub repository URL.

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook PCA.ipynb
```

Run the cells in sequence to load the dataset, standardize the features, apply PCA, inspect the variance ratios, and generate the visualization.

## 🚀 Applications of PCA

PCA is commonly used for:

- Data visualization
- Dimensionality reduction
- Feature extraction
- Noise reduction
- Exploratory data analysis
- Preparing high-dimensional data for machine learning models

## 🔮 Future Improvements

- Add a cumulative explained variance plot.
- Compare PCA results using different numbers of components.
- Add a legend identifying the three Iris species.
- Compare the original feature space with the PCA-transformed space.
- Evaluate classification performance before and after dimensionality reduction.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (Artificial Intelligence and Machine Learning)

## 📄 License

This project is intended for educational and learning purposes. Add a license file if you plan to distribute it under a specific open-source license.
