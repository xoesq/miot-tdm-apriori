#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Apriori
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


# Load transactions dataset
def readDataset(path):
    dataset = pd.read_csv(path, header = None)
    transactions = []
    for i in range(0, dataset.shape[0]):
        transactions.append(
            [str(dataset.values[i, j]) for j in range(0,dataset.shape[1])])
    return transactions

# Show pretty association results
def showResults(associations):
    results = []
    for item in list(associations):
        # first index of the inner list
        # Contains base item and add item
        pair = item[0] 
        items = [x for x in pair]
        value0 = str(items[0])
        if('nan' != value0 and len(items) >= 2):
            try:
                value1 = str(items[1])
            except:
                value1 = ''
                
            #second index of the inner list
            value2 = str(item[1])[:7]
        
            #third index of the list located at 0th
            #of the third index of the inner list
            value3 = str(item[2][0][2])[:7]
            value4 = str(item[2][0][3])[:7]
            
            rows = (value0, value1,value2,value3,value4)
            results.append(rows)
        
    labels = ['Item 1','Item 2','Support','Confidence','Lift']
    return pd.DataFrame.from_records(results, columns = labels)


# Read dataset
dataset = readDataset('dataset.csv')

# Train Apriori
association_rules = apriori(dataset, 
                            min_support=0.40, 
                            min_confidence=0.7,
                            min_lift=1, 
                            min_length=2)

# Show association rules
print(showResults(association_rules))







