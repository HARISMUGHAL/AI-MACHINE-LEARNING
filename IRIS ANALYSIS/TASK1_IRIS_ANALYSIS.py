import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

df = sn.load_dataset('iris')
print("dataset load successfully.\n")
print("Shape of the dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("dataset info :", df.info())
print("Summary:", df.describe())

###### SCATTER PLOT #############
plt.figure(figsize=(6, 6))
sn.scatterplot(data=df, x="sepal_length", y="petal_length", hue="species")
plt.title("sepal length vs petal length")
plt.show()

####### HISTOGRAMS (one per feature) ############
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

for feature in features:
    plt.figure(figsize=(6, 4))
    plt.hist(x=df[feature], bins=15, color='green', edgecolor='black')
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

########## BOX PLOT #########
plt.figure(figsize=(10, 6))
sn.boxplot(data=df)
plt.title("Box Plots of Iris Features")
plt.show()
