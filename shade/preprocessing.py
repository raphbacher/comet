# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:59:23 2015

@author: raphael
"""

import function_Image
import scipy.signal as ssl
from math import *
import numpy as np

class Preprocessing():
    """

    """

    def __init__(self,cube,listSources,processedCube,params,paramsPreProcess):
        self.cube=cube
        self.listSources=listSources
        self.processedCube=processedCube
        self.params=params
        self.paramsPreProcess=paramsPreProcess

    def processSrc(self,):
        for i,src in enumerate(self.listSources): 
            if self.paramsPreProcess.forceProcess == False:
                if 'PROCESS_CUBE' in src.cubes.keys():                                
                    continue
            
            if self.params.SW is not None:
                if self.params.sim ==False:
                    center=src.cubes['MUSE_CUBE'].wcs.sky2pix([src.dec,src.ra])[0].astype(int)
                else:
                    #if sim is True source is supposed centered
                    center=[src.cubes['MUSE_CUBE'].shape[1]/2,src.cubes['MUSE_CUBE'].shape[2]/2]
                data=src.cubes['MUSE_CUBE'][:,max(center[0]-self.params.SW,0):min(center[0]+self.params.SW+1,src.cubes['MUSE_CUBE'].shape[1]), \
                    max(center[1]-self.params.SW,0):min(center[1]+self.params.SW+1,src.cubes['MUSE_CUBE'].shape[2])]
            else:
                data=src.cubes['MUSE_CUBE']
            if self.paramsPreProcess.unmask==True:
                data.data[:]=data.data.filled(np.nanmedian(data.data))
            if self.params.sim == False :
                lmbda=int(data.wave.pixel(src.lines['LBDA_OBS'][src.lines['LINE']=='LYA'][0]))+self.paramsPreProcess.shiftLambdaDetection
            else:
                lmbda=data.shape[0]/2
            
            fsf=self.params.fsf[max(lmbda-self.params.LW-self.params.lmbdaShift,0):min(lmbda+self.params.LW+self.params.lmbdaShift+1,self.params.fsf.shape[0])]
            #Process on a spectral slice that assure that for each value in the [-LW:LW] zone of interest, there are
            #at least 2*windowRC+1 points and so the removeContinuum method will work as expected
            dataRC=self.removeContinuum(data[max(lmbda-self.params.LW-self.paramsPreProcess.windowRC-self.params.lmbdaShift,0):min(lmbda+self.params.LW+self.paramsPreProcess.windowRC+self.params.lmbdaShift+1,data.shape[0]),:,:])

            lmbda=dataRC.shape[0]/2                    
            dataRC=dataRC[max(lmbda-self.params.LW-self.params.lmbdaShift,0):min(lmbda+self.params.LW+self.params.lmbdaShift+1,dataRC.shape[0]),:,:]
            
            
            dataMF=self.matchedFilterFSF(dataRC,fsf)
            src.cubes['PROCESS_CUBE']=dataMF
            
    def processSrcWithCube(self):
        self.processCube()
        for src in self.listSources():
            try:
                src.cubes['PROCESS_CUBE']
                continue
            except:
                center=self.cubeProcessed.wcs.sky2pix([source.dec,source.ra])[0]
                lmbda=self.cubeProcessed.wave.pixel(src.lines['LBDA_OBS'][src.lines['LINE']=='LYA'][0])
                processedData=self.cubeProcessed[max(lmbda-self.params.LW-self.params.lmbdaShift,0):min(lmbda+self.params.LW+self.params.lmbdaShift+1,dataRC.shape[0]), \
                    max(center[0]-self.params.SW,0):min(center[0]+self.params.SW+1,self.cubeProcessed.shape[1]), \
                    max(center[1]-self.params.SW,0):min(center[1]+self.params.SW+1,self.cubeProcessed.shape[2])]
                src.add_cube(processedData, 'PROCESS_CUBE')
            

    def processCube(self):
        if self.processedCube is None:
            if self.paramsPreProcess.lmbdaMax is None:
                self.paramsPreProcess.lmbdaMax=self.cube.shape[0]
            if self.params.SW is not None:
                data=self.cube[self.paramsPreProcess.lmbdaMin:self.paramsPreProcess.lmbdaMax, center[0]-self.params.SW:center[0]+self.params.SW+1,center[1]-self.params.SW:center[1]+self.params.SW+1]
            else:            
                data=self.cube[self.paramsPreProcess.lmbdaMin:self.paramsPreProcess.lmbdaMax,:,:]
            fsf=self.params.fsf[self.paramsPreProcess.lmbdaMin:self.paramsPreProcess.lmbdaMax]
            dataRC=self.removeContinuum(data)
            dataMF=self.matchedFilterFSF(dataRC,fsf)
            self.cubeProcessed=dataMF


    
    def removeContinuum(self,cube):
        cubeContinuRemoved=cube.copy()        
        if self.paramsPreProcess.methodRC == 'medfilt':        
            cubeContinuRemoved.data=cube.data-ssl.medfilt(cube.data,[self.paramsPreProcess.windowRC,1,1])
        return cubeContinuRemoved

    def matchedFilterFSF(self,cubeContinuRemoved,fsf,):
        
        cubeContinuRemoved.data = cubeContinuRemoved.data/np.sqrt(cubeContinuRemoved.var)
        f = function_Image.fine_clipping2
        #cubeMF = cubeContinuRemoved.loop_ima_multiprocessing(f, cpu = 6, verbose = False, \
        #    Pmin=self.paramsPreProcess.Pmin, Pmax=self.paramsPreProcess.Pmax, Qmin=self.paramsPreProcess.Qmin, Qmax=self.paramsPreProcess.Qmax) #
        cubeMF=cubeContinuRemoved
        if self.paramsPreProcess.spatialCentering==True:        
            for i in xrange(cubeMF.shape[0]):
                cubeMF[i,:,:]=f(cubeMF[i,:,:],Pmin=self.paramsPreProcess.Pmin, Pmax=self.paramsPreProcess.Pmax, Qmin=self.paramsPreProcess.Qmin, Qmax=self.paramsPreProcess.Qmax,unmask=self.paramsPreProcess.unmask)
        #cubeMF = cubeContinuRemoved
        # ---- Matched Filter (MF) ---- #
        if self.paramsPreProcess.FSFConvol==True:
            for i in xrange(cubeMF.shape[0]):
                cubeMF[i,:,:]=function_Image.Image_conv(cubeMF[i,:,:],fsf[i],self.paramsPreProcess.unmask)
        
        #cubeMF=cubeMF.loop_ima_multiprocessing(f, cpu = 6, verbose = False,Pmin=self.paramsPreProcess.Pmin, Pmax=self.paramsPreProcess.Pmax, \
        #Qmin=self.paramsPreProcess.Qmin, Qmax=self.paramsPreProcess.Qmax)
        
        return cubeMF
    
    
