# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:50:27 2015

@author: raphael
"""
import numpy as np
import function_Image
from scipy.stats.mstats import mquantiles

class EmpFunc2():
    """"
    Define an empirical cumulative distribution function with a set of values.
    """
    
    def __init__(self,values):
        """
        Constructor of an empirical distribution. 
        Param: ndarray values :The sample of the empirical values is 
        """
        self.n=float(values.size)
        self.values=np.sort(values.flatten())
        self.quantiles=mquantiles(values,prob=np.arange(self.n)/self.n,alphap=1/3., betap=1./3,)
            
    def calc(self,newValue):
        """
        Compute empirical cumulative distribution function values for one sample as input        
        """
        idx=np.searchsorted(self.quantiles, newValue, side="left")
        if idx>=self.n-1:
            return idx/self.n
        if np.abs(newValue - self.quantiles[idx-1]) < np.abs(newValue - self.quantiles[idx]):
            return (idx-1)/self.n
        else:
            return idx/self.n
         
        
    def calcECDF(self,arr):
        """
        Compute empirical cumulative distribution function values for an array as input

        Param: ndarray arr

        Return: ndarray res, same shape as arr
        """
        res=np.zeros_like(arr)
        for index, value in np.ndenumerate(arr):
            res[index]=self.calc(value)
        return res


class EmpFunc():
    """"
    Define an empirical cumulative distribution function with a set of values.
    """
    
    def __init__(self,values):
        """
        Constructor of an empirical distribution. 
        Param: ndarray values :The sample of the empirical values is 
        """
        self.n=float(values.size)
        self.values=values
            
    def calc(self,newValue):
        """
        Compute empirical cumulative distribution function values for one sample as input        
        """
        return np.sum(self.values<=newValue)/self.n
        
    def calcECDF(self,arr):
        """
        Compute empirical cumulative distribution function values for an array as input

        Param: ndarray arr

        Return: ndarray res, same shape as arr
        """
        res=np.zeros_like(arr)
        for index, value in np.ndenumerate(arr):
            res[index]=self.calc(value)
        return res
        
def calcPvalue(corrMap,H0=None,correct=True):
    """
    Compute map of pvalues from a correlation 3D array.

    Param:3D array corrMap, dim Number of shifted references x Spatial x Spatial.
    Param: array H0 of H_0 samples for calibration 
    Each slice represents the correlation between the pixels of the datacube and one particular target spectrum.
    
    Return: 2D array, pvalues of the test of the max of correlation
    """
    listM_Corr=[]
    listS_Corr=[]
    listMu_Corr=[]
    
    #For each correlation map (resulting from the correlation with one reference), we estimate 
    #the parameter of a student distribution
    
    for k in xrange(len(corrMap)):
        #res=function_Image.getStudentParam(corrMap[k],runLikelihood=True)
        res=function_Image.getParamNoise(corrMap[k])
        listM_Corr.append(res[0])
#        listS_Corr.append(res[1])
#        listMu_Corr.append(res[2])

    # with the estimated mean parameter we center each map to ensure that the null hypothesis is
    #symmetric.
    resCorrCentr=np.zeros_like(corrMap)
    for k in xrange(len(resCorrCentr)):
        resCorrCentr[k]=corrMap[k]-listM_Corr[k]
        #resCorrCentr[k]=corrMap[k]
        
    #We can then compute the empirical distribution of the opposite of the mins of correlation.
    minVal=np.min(resCorrCentr,axis=0)
    maxVal=np.max(resCorrCentr,axis=0)
    mMax=np.median([maxVal,-minVal])
    if H0 is None:
        if correct==True:        
            empFuncCorr1=EmpFunc2(maxVal)
            empFuncCorr2=EmpFunc2(-minVal[minVal<=-mMax])
        else:
            empFuncCorr=EmpFunc2(-minVal[minVal<-np.min(maxVal)])
        #empFuncCorr=EmpFunc(-minVal)
    else:
        empFuncCorr=EmpFunc2(H0)
    #Finally we can compute the pvalues of the test of the max of correlation using this empirical function.
    if H0 is None:
        if correct==True:        
            pvalCorr=np.zeros_like(maxVal)
            pvalCorr[maxVal<mMax]=(1-empFuncCorr1.calcECDF(maxVal[maxVal<mMax]))
            pvalCorr[maxVal>=mMax]=(1-empFuncCorr2.calcECDF(maxVal[maxVal>=mMax])/2.-0.5)
        else:
            pvalCorr = (1-empFuncCorr.calcECDF(maxVal))
    else:
        pvalCorr = (1-empFuncCorr.calcECDF(maxVal))
    
    return pvalCorr


def connexFDR(corrArr,q):
    maxVal=np.max(corrArr,axis=0)
    minVal=-np.min(corrArr,axis=0)
    minValF=minVal.flatten()
    maxValF=maxVal.flatten()
    argsortPval=np.argsort(maxVal,axis=None)
    argsortPval=argsortPval[::-1]
    
    
    pvalSorted=maxVal.flatten()[argsortPval]
    
    setDetected=set([argsortPval[0]])
    listDetected=np.array(list(setDetected))
    setNeighbors=set(getNeighbors(setDetected,maxVal))
    listNeighbors=np.array(list(setNeighbors))
    oldDetected=np.array([0])
    niter=0

    while set(oldDetected)!=set(listDetected) and niter<50:
        ll=[]
        oldDetected=listDetected.copy()
        for a in listNeighbors:
            ll.append(np.nonzero(argsortPval==a)[0])
        indices=np.sort(np.array(ll),axis=0)
        k=findNewDetection(listNeighbors,pvalSorted,argsortPval,maxValF,minValF,q,indices)
        p=pvalSorted[indices][k]
        newDetections=np.array(listNeighbors)[np.array(maxValF[listNeighbors]>=p)]
        setDetected=set([x for x in newDetections])
        listDetected=np.array(list(setDetected))
        #setNeighbors=setNeighbors.union(getNeighbors(newDetections,maxVal))
        setNeighbors=getNeighbors(newDetections,maxVal)
        listNeighbors=np.array(list(setNeighbors))
        niter=niter+1
    mask=getMask(oldDetected,maxVal)
    return mask

def getNeighbors(k_set,arr):
    neighbors=set()
    for k in list(k_set):
        i,j=np.unravel_index(k,arr.shape)
        imax=np.maximum(i-1,0)
        imin=np.minimum(i+1,arr.shape[0]-1)
        jmax=np.maximum(j-1,0)
        jmin=np.minimum(j+1,arr.shape[1]-1)
        neighbors=neighbors.union(set(np.ravel_multi_index([[imin,imin,imin,i,i,i,imax,imax,imax],[jmin,j,jmax,jmin,j,jmax,jmin,j,jmax]],arr.shape)))
    return neighbors
    
def findNewDetection(listNeighbors,pvalSorted,argsortPval,maxValF,minValF,q,indices):    
    try:
        k=[np.sum(minValF[listNeighbors]>=p)/np.maximum(1,float(np.sum(maxValF[listNeighbors]>=p)))<=q for p in pvalSorted[indices]].index(False)    
    except:
        k=argmin(pvalSorted[indices])
    return k

def getMask(listDetected,arr):
    mask=np.empty_like(arr,dtype=bool)
    mask[:]=False
    for x in list(listDetected):
        mask[unravel_index(x,arr.shape)]=True
    return mask