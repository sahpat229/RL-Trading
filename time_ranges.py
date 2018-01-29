import csv

file = open('./data/csvs/data.csv')
reader = csv.reader(file)
next(reader)

time_ranges = {}
for line in reader:
    coin = line[1]
    if coin not in time_ranges:
        time_ranges[coin] = [int(line[0]), None]
    else:
        time_ranges[coin][1] = int(line[0])

print(time_ranges)
print(max([time_ranges[i][0] for i in time_ranges]))
file.close()