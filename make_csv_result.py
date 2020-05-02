import csv
import glob
import re
import os

seeds = [12, 41, 95]
a_list = [8, 7, 9]

d = '.'

cur_dir = os.getcwd()

file_name = []

for name in glob.glob('*.csv'):
    if re.match('(\w+)_test_result.csv', name):
        print(0, name)
        file_name.append(name)
        if re.match('(\d+)_(True|False)_SBM_(CLUSTER|PATTERN)_(a\d+|w\d+)_*', name):
            name_list = name.split("_")
            seed = name_list[0]
            residual = name_list[1]
            dataset = name_list[3]
            smoothing_name = name_list[4]
            model_name = name_list[5]
            with open(name) as f:
                csvreader = csv.reader(f,  delimiter=',')
                next(csvreader)
                accuracy = float(csvreader[0][0])



def read_csv(cur_dir, fileName):
    return

