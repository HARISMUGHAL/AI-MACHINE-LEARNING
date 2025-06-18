# Task 1: Iris Dataset Analysis 🌸

## 📋 Objective

This task involves exploring the famous Iris dataset to understand the distribution and relationships between its features using various visualization techniques.

## 📁 Dataset

The Iris dataset is loaded using Seaborn’s built-in dataset loader:
- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Target:
  - Species (Setosa, Versicolor, Virginica)

## 🧪 What This Script Does

### 1. Data Loading & Summary
- Loads the dataset via `seaborn.load_dataset('iris')`.
- Displays the shape, column names, data types, and summary statistics.

### 2. Visual Exploratory Data Analysis (EDA)

#### 🔷 Scatter Plot
- **Purpose**: Visualize relationships between sepal length and petal length across species.
- **Tool Used**: `seaborn.scatterplot`

#### 📊 Histograms
- **Purpose**: Understand the frequency distribution of each feature.
- **Features**: All four numerical columns.
- **Tool Used**: `matplotlib.pyplot.hist`

#### 📦 Box Plot
- **Purpose**: Spot outliers and compare feature distributions.
- **Tool Used**: `seaborn.boxplot`

## 📌 Libraries Used

- `pandas` – for data structure and manipulation
- `seaborn` – for loading dataset and plotting
- `matplotlib.pyplot` – for detailed custom plots

## 🧠 Skills Practiced

- Exploratory Data Analysis (EDA)
- Data visualization
- Working with built-in datasets
- Plotting histograms, scatter plots, and boxplots

## 📈 Sample Output

- Scatter plot showing species separation
- Histograms for all features
- Boxplot comparing feature spread

---

✅ **Note**: This script is ideal for beginners learning data visualization and understanding the relationship between numerical features in a multi-class classification dataset.
