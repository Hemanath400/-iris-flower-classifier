import pandas as pd
import matplotlib.pyplot as plt

print("🎨 Creating Visualization...")
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

df=pd.read_csv(url)

plt.figure(figsize=(10,9))

color={'setosa':'red','versicolor':'green','virginica': 'blue'}

for species,group in df.groupby('species'):
    plt.scatter(group['sepal_length'],group['sepal_width'],
                color=color[species],
                alpha=0.7,
                label=species)

plt.xlabel('sepal Lenght in (cm)')
plt.ylabel('sepal width in (cm)')
plt.title("iris flower sepal length v sepal widht")
plt.legend()
plt.savefig('iris plot.png')
print("✅ Plot saved as 'iris_plot.png'")