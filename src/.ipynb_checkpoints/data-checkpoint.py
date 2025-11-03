import pandas as pd
import numpy as np

dataset1 = pd.read_csv("student-mat.csv")#reading the csv file and converting it to pandas dataframe
dataset2 = pd.read_csv("student-por.csv")
#checking if the number of columns and the column name match in two dataframes or not
set1 = set(dataset1.columns)
set2 = set(dataset2.columns)
print(f"dataset 1 size:{dataset1.shape}")
print(f"dataset 1 size:{dataset2.shape}")
print("Columns only in dataset1:", set1 - set2)#no columns found which is only present in dataset1
print("Columns only in dataset2:", set2 - set1)#no columns found which is only present in dataset2
print("Common columns:", set1 & set2)# all of the columns match in both of the dataframes
mergeddf = pd.concat([dataset1, dataset2], ignore_index=True)#merging two dataframes
print(mergeddf.shape)# there are total 1044 rows and 33 columns in the new merged dataframe
