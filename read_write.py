from python_clustering import load_dataset


df = load_dataset.read_aiff('./datasets/s-set1.arff')
print(df['CLASS'])


