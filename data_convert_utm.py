import numpy as np
from datetime import datetime as d

quake_strings = []
with open('tremor_events_utm.csv', 'r') as data:
    for line in data:
        if len(line) > 1:
            quake_strings.append(line.split(','))

quake_data = []

for row in quake_strings:
    if row[0][0].isdigit():
        date = d.strptime(row[3].rstrip(), '%Y-%m-%d %H:%M:%S').timestamp()
        lat = float(row[4])
        log = float(row[5])
        depth = float(row[2])
        quake_data.append([date, lat, log, depth])

np.savetxt('earthquake_formatted_data_utm.txt', quake_data, fmt="%.7f", header="time lat long depth")
np.savetxt('earthquake_formatted_data_partial_utm.txt', quake_data[-1000:], fmt="%.7f", header="time lat long depth")