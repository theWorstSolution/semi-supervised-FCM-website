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
import random

from io import StringIO
from io import BytesIO

import json
import base64

import time

m = 0
epsilon = 0
inputTable = None
nCenters = 0
inputFile = None
maxIters = 1000

# Create your views here.
def home_view(request,*args,**kwargs):
    
    context = {}
    return render(request,"home.html",context)

def clustering1():
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

def calculateDistance(point, center):
    distance = 0.0
    for i in range(point.size):
        distance += pow(point[i] - center[i], 2)
    distance = pow(distance, 0.5)
    return distance
#print(calculateDistance(np.array([4,5]),np.array([1,1])))

def sfcmClustering(points, nCenters, initU,m=2, maxError=0.005, maxIter=1000, init=None):
    #random.seed(5
    #so diem
    print("nCenters")
    print(nCenters)
    nPoints = points.shape[0]
    print("nPoints")
    print(nPoints)
    #so chieu cua moi diem
    dim = points.shape[1]
    print("dim")
    print(dim)
    
    centers = np.array([[]])
    centers = centers.reshape(-1,dim)
    
    u = np.zeros((nPoints, nCenters))
    
    error = 0
    
    #chon random nCenters diem lam tam
    randIndexList = random.sample(range(nPoints), nCenters)
    for randIndex in randIndexList:
        centers = np.vstack((centers, points[randIndex] * 0.9))
    #print(centers)
    error = 9999
    iters = 0
    while(error > maxError and iters < maxIter):
        #tinh membership
        error = 0
        for i in range(nCenters):
            for k in range(nPoints):
                d_ki = calculateDistance(points[k], centers[i])
                oldU_ki = u[k][i]
                u[k][i] = 0
                sumInitU = 0
                denominator = 0
                for j in range(nCenters):
                    sumInitU += initU[k][i]
                    denominator += pow(1 / calculateDistance(points[k], centers[j]), 1 / (m-1))
                u[k][i] = initU[k][i] + (1 - sumInitU)*pow(1 / calculateDistance(points[k], centers[i]), 1 / (m-1)) / denominator
                error += abs(u[k][i] - oldU_ki)
        #tinh diem moi
        for i in range(nCenters):
            numerator = 0
            denominator = 0
            for k in range(nPoints):
                numerator += pow(abs(u[k][i] - initU[k][i]), m) * points[k]
                denominator += pow(abs(u[k][i] - initU[k][i]), m)
            centers[i] = numerator / denominator
        iters += 1
    return u, centers, error, iters
    
class Cluster:
    pass

def handleInput(request,*args,**kwargs):
    global m
    global epsilon
    global inputTable
    global inputFile
    global nCenters
    global maxIters
    #start_time = time.time()
    if request.method == 'POST':
        inputFile = request.FILES['dataset'].read()
        dict = request.POST.dict()
        m = float(dict['m-'])
        epsilon = float(dict['epsilon-'])
        nCenters = int(dict["nCenters-"])
        maxIters = int(dict["maxIters-"])
        #maxCluster = int(dict["maxCluster-"])
        #print(minCluster)
        #print(maxCluster)
        #print(file.decode("utf-8"))
        #print(type(file))
        inputTable = pd.read_csv(BytesIO(inputFile))
        inputTableHtml = inputTable.to_html()
        #return HttpResponse("previewTable.html", {"inputTableHtml": inputTableHtml})
        context = {"inputTableHtml": inputTableHtml}
        previewTable = render_to_string("previewInput.html", context)
        return JsonResponse({'previewTable': previewTable, "nCenters": nCenters})
    return HttpResponse("")
    
def clustering(request,*args,**kwargs):
    global m
    global epsilon
    global inputTable
    global inputFile
    global nCenters
    global maxIters
    start_time = time.time()
    if request.method == 'POST':
         
        #file = request.FILES['dataset'].read()
        #dict = request.POST.dict()
        #initU = 
        initU_list = request.POST.getlist('initU[]')
        #print(arr)
        print(type(initU_list))
        initU = np.asarray(initU_list, dtype=np.float32)
        initU = initU.reshape(-1, nCenters)
        print("initU")
        print(initU)
        data = inputTable.to_numpy()
        print("data")
        print(data)
        u, centers, error, iters = sfcmClustering(data, nCenters, initU, m, epsilon, maxIter=maxIters)
        efficiency = 0
        clusters = []
        # w, h = np.shape(center)
        # print(w)
        # print(h)
        #print("shape: ")
        #print(np.shape(center))
        #clusters = [[None] * w for i in range(h)]
        cluster_membership = np.argmax(u, axis=1)
        print("cluster_membership")
        print(cluster_membership)
        for i in range(nCenters):
            print(i)
            #i = i + 1
            #clusters[i].center = temp.append(pd.DataFrame(center[i])).to_html()
            temp = Cluster()
            print("centers[i]")
            print(centers[i])
            print("centers[i].shape")
            print(centers[i].shape)
            temp.center = pd.DataFrame(centers[i].reshape(1,-1), columns=list(inputTable)).to_html()
            temp.points = inputTable[cluster_membership == i].to_html()
            clusters.append(temp)
            # context[f"tamCum{i}"] = tamCum[i].to_html()
            # context[f"cum{i}"] = cum[i].to_html()
        
        context = {"clusters": clusters}
        executionTime = "%s seconds" % (time.time() - start_time)
        context1 = {"totalIter": iters, "efficiency": efficiency, "executionTime": executionTime, "nCenters": nCenters, "error": error}
        result = render_to_string('result.html', context)
        information = render_to_string('information.html', context1)
        return JsonResponse({'result': result,'information': information})