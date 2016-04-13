# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:59:18 2015

@author: raphael
"""


from array_tools import normArr
import proba_tools
import numpy as np

            
class Detection:
    
    
    def __init__(self,listSources,params,paramsPreProcess,paramsDetection):
        self.params=params
        self.paramsPreProcess=paramsPreProcess
        self.paramsDetection=paramsDetection
        self.listSources=listSources
        self.listPvalMap=[]
        self.listCorrArr=[]
        self.listIndexMap=[]
        
        
        
    def detect(self):
        """
        Compute the detection test. At this point sources must contain a PROCESS_CUBE 
        of shape [LW-lmbda:LW+lmbda,center[0]-SW:center[0]+SW,center[1]-SW:center[1]+SW]
        or just [LW-lmbda:LW+lmbda,:,:] if SW is None.
        """
        for i,src in enumerate(self.listSources):
            try:
                src.cubes['PROCESS_CUBE']
            except:
                print "Warning : No cube named 'PROCESS_CUBE' in source %s. MUSE_CUBE will be used"%src.ID
                src.cubes['PROCESS_CUBE']=src.cubes['MUSE_CUBE']
            
            #Compute the 3D array of correlations between the spaxel map and a list of shifted target spectra
            #Dims of an element of listCorrMap : Number of target spectra x Spatial x Spatial
            self.listCorrArr.append(self.getCorrMap(src))
            
            #Then compute the 2D map of pvalues corresponding to the test of the maximal 
            #correlation between each spaxel and the list of target spectra
            #The maps are saved as mpdaf Images.
            Im=src.cubes['PROCESS_CUBE'][0,:,:].clone()
            Im.data=proba_tools.calcPvalue(self.listCorrArr[-1])
            self.listPvalMap.append(Im)
            Im2=src.cubes['PROCESS_CUBE'][0,:,:].clone()
            Im2.data=np.argmax(self.listCorrArr[-1],axis=0)
            self.listIndexMap.append(Im2)
        return self.listPvalMap,self.listIndexMap
                
    def getCorrMap(self,source):
        """
        Get the correlation map between pixels of the cube of a source object and 
        a list of target spectra.
        """
        zone=source.cubes['PROCESS_CUBE'].data
        lmbda=zone.shape[0]/2
        if self.params.sim == False:
            refPos=source.cubes['PROCESS_CUBE'].wcs.sky2pix([source.dec,source.ra])[0].astype(int)
        else:
            #for simulated sources sources are always supposed centered in the cube.
            refPos=[zone.shape[1]/2,zone.shape[2]/2]
        
        if self.params.SW is not None: # Resize zone of study accordingly
            zone=zone[:,max(refPos[0]-self.params.SW,0):min(refPos[0]+self.params.SW+1, \
            zone.shape[1]),max(refPos[1]-self.params.SW,0):min(refPos[1]+self.params.SW+1,zone.shape[2])]
            refPos=[zone.shape[1]/2,zone.shape[2]/2]
        
        zoneLarge=zone[max(lmbda-self.params.LW-self.params.lmbdaShift,0):min(lmbda+self.params.LW+self.params.lmbdaShift+1,zone.shape[0]),:,:]
        zoneCentr=zone[max(lmbda-self.params.LW+1,0):min(lmbda+self.params.LW,zone.shape[0]),:,:] #it is necessary to center between -LW+1 and LW (so a width of 2LW-1 instead of classical 2LW+1) to allow a shift of -lmbdashift:lmbdashift+1
        
        
        #centering (or not)    
        if self.paramsDetection.centering=='all':
            meanIm=np.mean(zoneCentr,axis=0)
            for i in xrange(zoneCentr.shape[0]):
                zoneCentr[i,:,:]=zoneCentr[i,:,:]-meanIm
        
        #Compute the target spectrum by averaging pixels around the center of the galaxy, then create a list of shifted versions.
        if self.paramsDetection.listRef is None:        
            listRef=[(np.mean(np.mean( \
                zoneLarge[:,refPos[0]-self.paramsDetection.windowRef:refPos[0]+self.paramsDetection.windowRef+1, \
                refPos[1]-self.paramsDetection.windowRef:refPos[1]+self.paramsDetection.windowRef+1],axis=2),axis=1))[k+1:zoneCentr.shape[0]+k+1] for k in xrange(zoneLarge.shape[0]-zoneCentr.shape[0]-1)]
        else:
            listRef=self.paramsDetection.listRef
        if (self.paramsDetection.centering=='ref') or (self.paramsDetection.centering=='all'):
            for l,ref in enumerate(listRef):
                listRef[l]=ref-np.mean(ref)
        
        #normalize spectra(or not)  
        if self.paramsDetection.norm == True:
            zoneNorm=normArr(zoneCentr)
        else:
            zoneNorm=zoneCentr
        
        #normalize target spectra (always)
        for l,ref in enumerate(listRef):
            listRef[l]=ref/np.sqrt(np.sum(ref**2))        
        
        self.listRef=listRef
        #Compute dot product between data and the list of referenced spectra
        res=np.zeros((len(listRef),zoneNorm.shape[1],zoneNorm.shape[2]))        
        for k in xrange(len(listRef)):
            for i in xrange(res.shape[1]):
                for j in xrange(res.shape[2]):
                    res[k,i,j]=np.dot(zoneNorm[:,i,j],listRef[k])        
        
        return res                