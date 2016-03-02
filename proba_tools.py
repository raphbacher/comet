# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:50:27 2015

@author: raphael
"""
import numpy as np
import function_Image

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
        
def calcPvalue(corrMap):
    """
    Compute map of pvalues from a correlation 3D array.

    Param:3D array corrMap, dim Number of shifted references x Spatial x Spatial .
    Each slice represents the correlation between the pixels of the datacube and one particular target spectrum.
    
    Return: 2D array, pvalues of the test of the max of correlation
    """
    listM_Corr=[]
    listS_Corr=[]
    listMu_Corr=[]
    
    #For each correlation map (resulting from the correlation with one reference), we estimate 
    #the parameter of a student distribution
    for k in xrange(len(corrMap)):
        res=function_Image.getStudentParam(corrMap[k],runLikelihood=True)
        listM_Corr.append(res[0])
        listS_Corr.append(res[1])
        listMu_Corr.append(res[2])
    # with the estimated mean parameter we center each map to ensure that the null hypothesis is
    #symmetric.
    resCorrCentr=np.zeros_like(corrMap)
    for k in xrange(len(resCorrCentr)):
        resCorrCentr[k]=corrMap[k]-listM_Corr[k]

    #We can then compute the empirical distribution of the opposite of the mins of correlation.
    empFuncCorr=EmpFunc(-np.min(resCorrCentr,axis=0))
    #Finally we can compute the pvalues of the test of the max of correlation using this empirical function.
    pvalCorr = (1-empFuncCorr.calcECDF(np.max(resCorrCentr,axis=0)))
    
    return pvalCorr