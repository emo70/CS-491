import sys
sys.path.append('./anaconda3/Lib/site-packages')
import c3d
import numpy as np
import math
import os
import pandas as pd
import csv

def getEMGFeature(File):
    r = c3d.Reader(open(File, 'rb'))
    EMGFeatures = list()
    EMGData = dict()
    frame_no = 1
    for _, _, analog in r.read_frames():
        for i in range(13, 29):
            if frame_no == 1:
                EMGData[i - 12] = list()
            EMGData[i - 12] += list(analog[i])
        frame_no += 1
    N = len(EMGData[1])
    for i in range(1, len(EMGData) + 1):
        EMGData[i] = np.array(EMGData[i])
    # 1. MAV, Mean Absolute Value of EMG Signal for 16 channels (Time based on the frame 0.008 sec)
    for i in range(len(EMGData)):
        EMGFeatures += [ np.sum(np.absolute(EMGData[1 + i])) / N ]
        
    # 2. Mean Absolute Value Slope (MAVS)
    # Same idea with the velocity, 10 bins, and each bin have 16 channels
    EMGDataBins = np.zeros(10 * len(EMGData))
    for i in range(len(EMGData)):
        bins = [x for x in np.array_split(EMGData[i+1], 10) if len(x) > 0] 
        EMGDataBins[i * len(bins)] = np.absolute(np.sum(bins[0])) / len(bins[0])
        for j in range(1, len(bins)):
            EMGDataBins[i * len(bins) + j] = np.absolute(np.sum(bins[j])) / len(bins[j])
            EMGDataBins[i * len(bins) + j] -= EMGDataBins[i * len(bins) + j - 1]
    EMGFeatures += list(EMGDataBins)
    
    # 3. Variance of EMG (VAR)
    for i in range(len(EMGData)):
        EMGFeatures += [ np.var(EMGData[i+1]) ]
    
    return EMGFeatures

def getVelocityNormFeature(File, pointLength = None):
    
    length = 80
    if pointLength is not None:
        length = pointLength
    r = c3d.Reader(open(File, 'rb'))
    i = 1
    data = dict()
    for _, points, _ in r.read_frames():
        data[i] = points
        i += 1
    
    velocity = dict()
    
    for i in range(2, len(data)):
        index = 0
        velocity[i - 2] = np.zeros(len(data[i]) * 3)
        for point, prevPoint in zip(data[i], data[i-1]):
            if point[3] != -1 and prevPoint[3] != -1:
                for j in range(3):
                    velocity[i - 2][index] = (point[j] - prevPoint[j]) / (2 * 0.008)
                    index += 1
            else: index += 3
    
    
    v_norm = dict()
    for i in range(len(velocity)):
        v_norm[i] = np.zeros(int(len(velocity[i]) / 3))
        v_index = 0
        for index in range(len(v_norm[i])):
            for j in range(3):
                v_norm[i][index] += velocity[i][v_index] * velocity[i][v_index]
                v_index += 1
        v_norm[i][index] = math.sqrt(v_norm[i][index])
        
    # Only take 10 frames as our features
    per_sample = len(v_norm) // 10

    avg_v_norm = [0] * (10 * int(length))
    for i in range(1, 11):
        idx = (i - 1) * per_sample 
        for j in range(per_sample):
            index = idx + j
            for k in range(len(v_norm[i])):
                avg_v_norm[(i - 1) * length + k] += v_norm[index][k]
        for k in range(len(v_norm[i])):
            avg_v_norm[(i - 1) * length + k] /= per_sample
            
    return avg_v_norm

def getTopKPoints(avg_v_norm, K=18):
    per_sample = len(avg_v_norm) // 10
    new_list = [0] * K * 10
    for i in range(10):
        v_index = per_sample * i
        for j in range(K):
            new_list[i * K + j] = avg_v_norm[v_index + j]
    return new_list

def appendFeatueToSCV(FileName, csv_writer, label, label2):
    try:
        avg_v_norm = getVelocityNormFeature(FileName,pointLength = None)
        features = getTopKPoints(avg_v_norm)
        features += getEMGFeature(FileName)
    except:
        print("Error for file: ", FileName)
        return
    csv_writer.writerow(features + [label, label2])



rootDir = 'C:/Users/emeka/Drive/CS 491'
CSVFileName = 'C:/Users/emeka/Drive/CS 491/InputData.csv'
with open(CSVFileName, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for dirName, subdirList, fileList in os.walk(rootDir):
        if "HEALTHY CONTROLS" in dirName and "EPDMS" in dirName:
            label = "Healthy"
            label2 = "Unaffected"
            print('Found directory: %s' % dirName)
            for fname in fileList:   
                if fname.endswith(".c3d") and "Reaching" in fname:
                    print(label, '\t%s' % dirName + "/" + fname)
                    appendFeatueToSCV(dirName + "/" + fname, csv_writer, label, label2)
        elif "PARKINSON_s PATIENTS" in dirName and "EPDMS" in dirName:
            label = "Patient"
            label2 = ""
            if "UE affected" in dirName:
                label2 = "affected"
            elif "UE unaffected" in dirName:
                label2 = "unaffected"
            print('Found directory: %s' % dirName)
            for fname in fileList:   
                if fname.endswith(".c3d") and "Reaching" in fname:
                    print(label, ' ', label2, '\t%s' % dirName + "/" + fname)
                    appendFeatueToSCV(dirName + "/" + fname, csv_writer, label, label2)