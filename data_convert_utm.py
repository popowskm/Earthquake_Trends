import numpy as np
from datetime import datetime as d

quake_strings = []
with open('data/tremor_events_utm.csv', 'r') as data:
    for line in data:
        if len(line) > 1:
            quake_strings.append(line.split(','))

quake_data = []

for row in quake_strings:
    if row[0][0].isdigit():
        date = d.strptime(row[3].rstrip(), '%Y-%m-%d %H:%M:%S').timestamp()
        x = float(row[4])
        y = float(row[5])
        quake_data.append([date, x, y])

np.savetxt('data/earthquake_formatted_data_utm.txt', quake_data, fmt="%.7f", header="time x y")
np.savetxt('data/earthquake_formatted_data_partial_utm.txt', quake_data[-1000:], fmt="%.7f", header="time x y")