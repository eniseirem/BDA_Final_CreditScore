import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


missing_values = ["n/a", "na", "--", " ?","?"]

data = pd.read_csv('data/Credit Score Dataset.txt', delimiter="\t", na_values=missing_values)

print(data.tail)
