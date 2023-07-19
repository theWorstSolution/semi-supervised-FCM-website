from django.shortcuts import render
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from django.template.loader import render_to_string

import matplotlib
import matplotlib.pyplot as plt
from io import StringIO
from io import BytesIO
import numpy as np
import skfuzzy as fuzz
import pandas as pd

from io import StringIO
from io import BytesIO

import json
import base64

import time

m = 0
epsilon = 0
table = None
minCluster = -1
maxCluster = 20
# Create your views here.
def home_view(request,*args,**kwargs):
    
    context = {}
    return render(request,"home.html",context)

def clustering():
    dataTable = table.to_numpy()
    data = dataTable.transpose()
    #print(data)
    clusterTrialInfos = []
    for nCenters in range(minCluster, maxCluster+1):
        # center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        #         data, nCenters, 2, error=0.005, maxiter=1000, init=None)
        center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data, nCenters, m, error=epsilon, maxiter=1000, init=None)
        temp = {"center": center, "u": u, "p": p, "fpc": fpc,}
        clusterTrialInfos.append(temp)

    # Plot assigned clusters, for each data point in training set

    optimize = max(clusterTrialInfos, key=lambda x:x['fpc'])
    cluster_membership = np.argmax(optimize["u"], axis=0)
    totalIter = optimize["p"]
    efficiency = optimize["fpc"]
    return cluster_membership, optimize["center"], totalIter, efficiency

class Cluster:
    pass

def handleInput(request,*args,**kwargs):
    global m
    global epsilon
    global table
    global minCluster
    global maxCluster
    start_time = time.time()
    if request.method == 'POST':
        file = request.FILES['dataset'].read()
        dict = request.POST.dict()
        m = float(dict['m-'])
        epsilon = float(dict['epsilon-'])
        minCluster = int(dict["minCluster-"])
        maxCluster = int(dict["maxCluster-"])
        print(minCluster)
        print(maxCluster)
        #print(file.decode("utf-8"))
        #print(type(file))
        table = pd.read_csv(BytesIO(file))
        #table.describe()
        #table.head()
        #print(table.head().to_html)
        cluster_membership, center, totalIter, efficiency = clustering()
        #center = center.transpose()
        print("center: ")
        print(center)
        nCenters = np.size(center, 0)
        print("nCenters: ")
        print(nCenters)
        #temp = pd.DataFrame(columns=table.columns)
        
        clusters = []
        # w, h = np.shape(center)
        # print(w)
        # print(h)
        #print("shape: ")
        #print(np.shape(center))
        #clusters = [[None] * w for i in range(h)]
        for i in range(nCenters):
            print(i)
            #i = i + 1
            #clusters[i].center = temp.append(pd.DataFrame(center[i])).to_html()
            temp = Cluster()
            temp.center = pd.DataFrame(center[i].reshape(1,-1), columns=list(table)).to_html()
            temp.point = table[cluster_membership == i].to_html()
            clusters.append(temp)
            # context[f"tamCum{i}"] = tamCum[i].to_html()
            # context[f"cum{i}"] = cum[i].to_html()
        
        context = {"clusters": clusters}
        executionTime = "%s seconds" % (time.time() - start_time)
        context1 = {"totalIter": totalIter, "efficiency": efficiency, "executionTime": executionTime, "nCenters": nCenters}
        result = render_to_string('result.html', context)
        information = render_to_string('information.html', context1)
        return JsonResponse({'result': result,'information': information})
    context = {}
    return HttpResponse("")