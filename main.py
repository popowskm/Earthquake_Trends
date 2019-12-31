import numpy as np
data = np.loadtxt('earthquake_formatted_data.txt',skiprows=1)
print(data[-1])