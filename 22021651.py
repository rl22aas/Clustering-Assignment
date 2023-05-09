#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 00:55:23 2023

@author: Rimsha
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def data_characteristics_rel(data):
    _, ax = plt.subplots(2, 2, figsize = (20, 12))
    ax = ax.flatten()
    sns.kdeplot(ax=ax[0], x=data['weight(kg)'], 
                y=data['hemoglobin'], hue='smoking',data=data);
    sns.scatterplot(ax = ax[1], x = "weight(kg)", y = "hemoglobin",
                    hue = "smoking", size = "gender", sizes=(20, 100), 
                    legend="full", data = data);

    sns.kdeplot(ax=ax[2], x=data['serum creatinine'], 
                y = data['triglyceride'], hue='smoking',data=data);
    sns.scatterplot(ax = ax[3], x = "waist(cm)", y = "hemoglobin",
                    hue = "smoking", size = "gender", sizes=(20, 100), legend="full",
                    data = data);
    

def hemo_chol_clustering(data):
    data_x = data['Cholesterol']
    data_y = data['hemoglobin']
    kmeans = KMeans(n_clusters=3, random_state=0)
    data['cluster'] = kmeans.fit_predict(data[['Cholesterol', 
                                                     'hemoglobin']])

    centroids = kmeans.cluster_centers_
    cen_x = [i[0] for i in centroids] 
    cen_y = [i[1] for i in centroids]

    data['cen_x'] = data.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
    data['cen_y'] = data.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
    # define and map colors

    colors = ['#DF2020', '#81DF20', '#FFF']
    data['c'] = data.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})

    fig, ax = plt.subplots()

    ax.scatter(data_x, data_y, 
                c=data.c, alpha = 0.6, s=10)
    ax.set_facecolor('black')
    plt.xlabel("Cholesterol")
    plt.ylabel("Hemoglobin")

    plt.scatter(data['cen_x'], data['cen_y'], 10, "purple", marker="d",)
    # b, m = polyfit(data_x, data_y, 1)
    # plt.plot(data_x, b + m * data_x, '--')
    
    # plt.plot([data_x.mean()]*2, [0,3e7], color='#ddd', lw=0.5, linestyle='--')
    # plt.plot([0,3e9], [data_y.mean()]*2, color='#ddd', lw=0.5, linestyle='--')
    
    # z = np.polyfit(data_x, data_y, 2)
    # p = np.poly1d(z)

    # ax.plot(data_x, p(data_x), label='Fit', linewidth=0.2)


def hemo_chol_distribution(data):
    sns.kdeplot(x=data['Cholesterol'],y=data['hemoglobin'],
                hue='smoking',data=data);
    
    


d = os.path.dirname(__file__)
smoking_df = pd.read_csv(os.path.join(d, 'smoking.csv'))

print(smoking_df)

plt.figure(figsize = [20, 10], clear = True, facecolor = "white")
sns.heatmap(smoking_df.corr(), cmap = "Set2", annot = True, linewidths = 1);

print(smoking_df.columns)

# pd.plotting.scatter_matrix(smoking_df[['age',
#                                       'height(cm)',
#                                       'weight(kg)',
#                                       'serum creatinine',
#                                       'triglyceride',
#                                       'waist(cm)']], 
#                             figsize=(10, 10), s=5, alpha=0.6)

# plt.figure(figsize = [8, 8], clear = True, facecolor = "#FFF")
# smoking_df["gender"].value_counts().plot.pie(explode = [0, 0.15], 
#                                        autopct='%1.2f%%', shadow = True);


# Show relation between data characteristics
# data_characteristics_rel(smoking_df)

hemo_chol_clustering(smoking_df)
hemo_chol_distribution(smoking_df)

plt.show()

























