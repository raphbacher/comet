# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:58:41 2015

@author: raphael
"""

import numpy as np
from mpdaf.sdetect.source import Source

class Postprocess():
    
    def __init__(self,cube,listSources,listPvalMap,listIndexMap,params,paramsPreProcess,paramsDetection,paramsPostProcess):
        self.cube=cube
        self.listSources=listSources
        self.listPvalMap=listPvalMap
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
                newSrc=Source(src.ID,src.ra,src.dec,self.origin)
                newSrc.add_cube(src.cubes['MUSE_CUBE'],'MUSE_CUBE')
                newSrc.add_cube(src.cubes['PROCESS_CUBE'],'PROCESS_CUBE')
                self.listResultSources.append(newSrc)                
            else:
                newSrc=src
            #if 
            if self.paramsPostProcess.resizeCube == True:
                newSrc.cubes['MUSE_CUBE']=self.resizeCube(self.cube,newSrc.cubes['PROCESS_CUBE'])

            maskAll=self.createBinMap(self.listPvalMap[i])
            maskGal=self.createBinMapGal(self.listPvalMap[i],src)
            maskHal=maskAll-maskGal
            halSpec=self.createHaloSpec(maskAll,maskGal,self.cube)
            galSpec=self.createGalSpec(maskGal,src)
            
            newSrc.add_image(self.listPvalMap[i], 'DET_STAT')
            newSrc.add_image(self.listIndexMap[i], 'DET_INDEX_ALL')            
            newSrc.add_image(maskGal, 'DET_BIN_GAL')
            newSrc.add_image(maskHal, 'DET_BIN_HAL')
            
            newSrc.add_image(maskAll, 'DET_BIN_ALL')
            newSrc.add_spectrum(halSpec,'SPEC_HAL')
            newSrc.add_spectrum(galSpec,'SPEC_GAL')
            newSrc.origin=self.origin+newSrc.cubes['MUSE_CUBE'].filename
            

    
    
    def corrPvalueBH(self,im):
        """
        Entrées: 
            im : l'ensemble des pvalues du test
            threshold: le seuil de FDR voulu
        Sortie:
            newThresold : le seuil à appliquer à l'ensemble des valeurs pour vérifier le FDR.
        """
        l=np.sort(im.data.flatten())
        k=len(l)-1
        while (l[k] > ((k+1)/float(len(l))*self.paramsPostProcess.threshold)) and k>0:
            k=k-1
        self.thresholdFDR=l[k]

        
    def createBinMap(self,Im):
        Im1=Im.copy()        
        if self.paramsPostProcess.FDR == True:            
            Im1.data=Im.data<self.corrPvalueBH(Im)
        else:
            Im1.data=Im.data<self.paramsPostProcess.threshold
        return Im1            
        
        
    def createBinMapGal(self,mask,src):
        """
        For now the galaxy is defined by the FSF
        """
        res=np.zeros_like(mask)
        center=src.cubes['PROCESS_CUBE'].wcs.sky2pix([src.dec,src.ra])[0]
        ll=min([7,center[0],mask.shape[0]-center[0]-1,center[1],mask.shape[1]-center[1]-1])
        print self.params.fsf.shape
        res[center[0]-ll:center[0]+ll+1,center[1]-ll:center[1]+ll+1]= \
            self.params.fsf[int(self.cube.wave.pixel(src.cubes['PROCESS_CUBE'].wave.coord(src.cubes['PROCESS_CUBE'].shape[0]/2)))][10-ll:10+ll+1,10-ll:10+ll+1]>0.01
        return res

        
    
    def createHaloSpec(self,maskHal,maskGal,src):
        res=np.mean(src.cubes['MUSE_DATA'].data[:,maskHal & ~maskGal.astype(bool)],axis=1)
        spe=src.cubes['MUSE_DATA'][:,0,0].clone()
        spe.data=res
        return spe
    
    def createGalSpec(self,maskGal,src):
        res=np.mean(src.cubes['MUSE_DATA'].data[:,maskGal.astype(bool)],axis=1)
        spe=src.cubes['MUSE_DATA'][:,0,0].clone()
        spe.data=res
        return spe
    
    def resizeCube(self,cubeToResize,cubeRef):
        A0=cubeToResize.wcs.pix2sky(cubeRef.wcs.sky2pix([0,0])[0])[0].astype(int)
        B1=cubeToResize.wcs.pix2sky(cubeRef.wcs.sky2pix([cubeRef.shape[1],cubeRef.shape[2]])[0])[0].astype(int)
        lmbda=int(cubeToResize.wave.pixel(cubeRef.wave.coord(cubeRef.shape[0]/2),nearest=True))
        LW=cubeRef.shape[0]/2
        return cubeToResize[lmbda-LW:lmbda+LW+1,A0[0]:B1[0],A0[1]:B1[1]]
        