# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:58:41 2015

@author: raphael
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mpdaf.sdetect.source import Source
from shade import qvalues


class Postprocess():

    def __init__(self,cube,listSources,listPvalMap,listIndexMap,params,paramsPreProcess,paramsDetection,paramsPostProcess):
        self.cube=cube
        self.listSources=listSources
        self.listPvalMap=listPvalMap
        self.listIndexMap=listIndexMap
        self.params=params
        self.paramsPreProcess=paramsPreProcess
        self.paramsDetection=paramsDetection
        self.paramsPostProcess=paramsPostProcess




    def createResultSources(self):
        """
        Create Sources objects with results of the detection
        Can include binary detection maps, spectra estimation ...
        """
        if self.paramsPostProcess.newSource==True:
            self.listResultSources=[]
        for i,src in enumerate(self.listSources):


            if self.paramsPostProcess.newSource==True:
                newSrc=Source.from_data(src.ID,src.ra,src.dec,self.params.origin+[src.cubes['MUSE_CUBE'].filename,src.header['CUBE_V']])
                newSrc.cubes['MUSE_CUBE']=src.cubes['MUSE_CUBE']
                newSrc.cubes['PROCESS_CUBE']= src.cubes['PROCESS_CUBE']
                self.listResultSources.append(newSrc)
            else:
                newSrc=src

            if self.paramsPostProcess.resizeCube == True:
                newSrc.cubes['MUSE_CUBE']=self.resizeCube(self.cube,newSrc.cubes['PROCESS_CUBE'])

            maskAll=self.createBinMap(self.listPvalMap[i])
            maskGal=self.createBinMapGal(self.listPvalMap[i],src)
            maskHal=maskAll-maskGal
            maskHal.data=maskHal.data.astype(np.int)
            halSpec=self.createHaloSpec(maskAll,maskGal,src)
            galSpec=self.createGalSpec(maskGal,src)

            newSrc.images['DET_STAT']=self.listPvalMap[i]
            if self.paramsPostProcess.qvalue==True:
                newSrc.images['DET_QSTAT']=self.createQvalMap(self.listPvalMap[i])
            newSrc.images['DET_INDEX_ALL']=self.listIndexMap[i]
            newSrc.images['DET_BIN_GAL']= maskGal
            newSrc.images['DET_BIN_HAL'] = maskHal
            maskAll.data = maskAll.data.astype(np.int)
            newSrc.images['DET_BIN_ALL'] =maskAll
            newSrc.spectra['SPEC_HAL'] = halSpec
            newSrc.spectra['SPEC_GAL'] = galSpec
            #newSrc.origin=tuple(self.params.origin+[newSrc.cubes['MUSE_CUBE'].filename])



    def corrPvalueBH(self,im,threshold):
        """
        Entrées:
            im : l'ensemble des pvalues du test
            threshold: le seuil de FDR voulu
        Sortie:
            newThresold : le seuil à appliquer à l'ensemble des valeurs pour vérifier le FDR.
        """
        l=np.sort(im.data.flatten())
        k=len(l)-1
        while (l[k] > ((k+1)/float(len(l))*threshold)) and k>0:
            k=k-1
        self.thresholdFDR=l[k]
        return self.thresholdFDR


    def createBinMap(self,Im):
        Im1=Im.copy()
        if self.paramsPostProcess.FDR == True:
            Im1.data=Im.data<self.corrPvalueBH(Im,self.paramsPostProcess.threshold)
        else:
            Im1.data=(Im.data<self.paramsPostProcess.threshold).astype(np.int)
        return Im1

    def createQvalMap(self,Im):
        Im1=Im.copy()
        Im1.data=qvalues.estimate(Im.data)
        return Im1

    def createBinMapGal(self,mask,src):
        """
        For now the galaxy is defined by the FSF
        """
        res=np.zeros(mask.shape)
        Im=mask.clone()
        center=src.cubes['PROCESS_CUBE'].wcs.sky2pix([src.dec,src.ra])[0].astype(int)
        ll=int(min([7,center[0],mask.shape[0]-center[0]-1,center[1],mask.shape[1]-center[1]-1]))
        if self.params.sim ==False:
            lmbda=int(src.cubes['MUSE_CUBE'].wave.pixel(src.cubes['PROCESS_CUBE'].wave.coord(src.cubes['PROCESS_CUBE'].shape[0]/2)))
        else:
            lmbda=20
        res[center[0]-ll:center[0]+ll+1,center[1]-ll:center[1]+ll+1]= \
            self.params.fsf[lmbda][10-ll:10+ll+1,10-ll:10+ll+1]>0.01
        Im.data=res.astype(np.int)
        return Im



    def createHaloSpec(self,maskHal,maskGal,src):
        res=np.mean(src.cubes['MUSE_CUBE'].data[:,maskHal.data.astype(bool) & ~maskGal.data.astype(bool)],axis=1)
        spe=src.cubes['MUSE_CUBE'][:,0,0].clone()
        spe.data=res
        return spe

    def createGalSpec(self,maskGal,src):
        res=np.mean(src.cubes['MUSE_CUBE'].data[:,maskGal.data.astype(bool)],axis=1)
        spe=src.cubes['MUSE_CUBE'][:,0,0].clone()
        spe.data=res
        return spe

    def resizeCube(self,cubeToResize,cubeRef):
        A0=cubeToResize.wcs.sky2pix(cubeRef.wcs.pix2sky([0,0])[0],nearest=True)[0].astype(int)
        B1=cubeToResize.wcs.sky2pix(cubeRef.wcs.pix2sky([cubeRef.shape[1],cubeRef.shape[2]])[0],nearest=True)[0].astype(int)
        lmbda=int(cubeToResize.wave.pixel(cubeRef.wave.coord(cubeRef.shape[0]/2),nearest=True))
        LW=cubeRef.shape[0]//2
        return cubeToResize[lmbda-LW:lmbda+LW+1,A0[0]:B1[0],A0[1]:B1[1]]

