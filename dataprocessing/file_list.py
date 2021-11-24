import os
import glob
import numpy as np
import pandas as pd
import csv

file_path = 'D:/0902_Thyroid/ThyroidSPECT Dataset/'

file_list = os.listdir(file_path)

f = open('thyroid_file_list.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
for f in file_list:
    print(f)
    wr.writerow([f])
f.close()
print(f'len = {len(file_list)}')
print(f'file_list = {file_list}')