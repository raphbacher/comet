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


def connexAggr(corrMap,q,core=None,returnNeighbors=False,w=1,coeff=1.2,seed=None):
    corrArr = corrMap-np.array([function_Image.getParamNoise(corrMap[k])[0] for k in xrange(len(corrMap))])[:,None,None]
    maxVal = np.max(corrArr, axis=0)
    minVal = -np.min(corrArr, axis=0)
    minValF = minVal.flatten()
    maxValF = maxVal.flatten()

    argsortPval = np.argsort(maxVal, axis=None)
    argsortPval = argsortPval[::-1]

    coreMask = np.zeros_like(maxVal, dtype=bool)

    if core is not None:
        if seed is not None:
            for s in seed:
                coreMask[s[0]-core:s[0]+core+1, s[1]-core:s[1]+core+1] = True
        else:
            coreMask[coreMask.shape[0]/2-core:coreMask.shape[0]/2+core+1,coreMask.shape[1]/2-core:coreMask.shape[1]/2+core+1]=True
        coreMask = coreMask.flatten()

        setAll = set([x for x in np.nonzero(coreMask)[0]])

    else:
        coreMask = coreMask.flatten()
        setAll = set([argsortPval[0]])
    listAll = np.array(list(setAll))
    setNeighbors=set(getNeighbors(setAll,maxVal,w=w)).difference(setAll)
    listNeighbors=np.array(list(setNeighbors))
    if core is not None:
        p=len(listAll)
    else:
        p = 1
    n = 0
    listAll_valid = None

    qq = 0
    niter = 0
    while qq < coeff*q or niter < 1./q+10:

        ll = []
        for a in listNeighbors:
            ll.append(np.nonzero(argsortPval == a)[0])

        try:
            k_=np.argmax(np.maximum(maxValF[listNeighbors],minValF[listNeighbors]))
        except:
            break
        k = listNeighbors[k_]
        setAll.update([k])
        setNeighbors = setNeighbors.union(getNeighbors(set([k]), maxVal, w))
        setNeighbors = setNeighbors.difference(setAll)
        listNeighbors = np.array(list(setNeighbors))
        listAll = np.array(list(setAll))
        if maxValF[k] < minValF[k]:
            n = n+1.
        elif maxValF[k] > minValF[k]:
            p = p+1.
        qq = (1+n)/np.maximum(p, 1)
        if qq <= q:
            listAll_valid = listAll.copy()
        niter = niter+1
    if listAll_valid is not None:
        listAll = listAll_valid
    else:
        mask = np.empty_like(maxVal, dtype=bool)
        mask[:] = False
        return mask
    try:

        listDetected = np.array(listAll)[np.array(maxValF[listAll] > minValF[listAll])]
        mask = getMask(listDetected, maxVal)
    except:
        mask = np.empty_like(maxVal, dtype=bool)
        mask[:] = False
    return mask


def getNeighbors(k_set, arr, w=1):
    neighbors = set()
    for k in list(k_set):
        i, j = np.unravel_index(k, arr.shape)
        imax = np.maximum(i-w, 0)
        imin = np.minimum(i+w, arr.shape[0]-1)
        jmax = np.maximum(j-w, 0)
        jmin = np.minimum(j+w, arr.shape[1]-1)
        x = []
        for l in range(imax, imin+1):
            x = x+[l]*(jmin-jmax+1)
        y = range(jmax, jmin+1)*(imin-imax+1)

        neighbors = neighbors.union(set(np.ravel_multi_index([x, y], arr.shape)))
    return neighbors


def getMask(listDetected, arr):
    mask = np.empty_like(arr, dtype=bool)
    mask[:] = False
    for x in list(listDetected):
        mask[np.unravel_index(x, arr.shape)] = True
    return mask
