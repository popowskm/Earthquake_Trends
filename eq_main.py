import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from datetime import datetime as d

quake_strings = []
with open('eq_data.txt', 'r') as data:
    for line in data:
        if len(line) > 1:
            quake_strings.append(line.split())

quake_data = []

for row in quake_strings:
    if row[0][0].isdigit():
        date = d.strptime(row[0]+row[1], '%Y-%m-%d%H:%M:%S').timestamp()
        lat = float(row[2])
        log = float(row[3])
        quake_data.append([date, lat, log])

print(quake_data[-1])