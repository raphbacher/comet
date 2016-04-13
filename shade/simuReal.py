# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:42:33 2015

@author: raphael
"""
from mpdaf.sdetect.source import Source
from mpdaf.obj import Cube,Image,WCS,WaveCoord



import numpy as np
import scipy.linalg as sl
import scipy.spatial.distance as dists
import scipy.stats as sst
from scipy.stats import multivariate_normal
from math import *
import utils

class SourceSim():
    """
    Build a source object with possible several point sources (with similar emission lines) 
    and linking "filaments".
    This class inheritates from the mpdaf Source and can be process like a real source object by the detection algorithms
    Overloads attributes are only used to ease the generation of the simulated data and keep most of 
    the building process information.
    """
    def __init__(self,shape=(41,101,101),lmbda=20,noise=None, \
        spectraSourcesLmbda=None,spectraSourcesWidth=5,listCenter=None, \
        listRadius=None,link=None,intens=0.2,rho=0,variation=0):

        """
        Param:
        Param:
        Param:
        """

        self.shape=shape
        self.lmbda=lmbda
        self.data=np.zeros(shape)
        self.listCenter=listCenter
        self.center=listCenter[0]
        self.listRadius=listRadius        
        self.intens=intens
        self.link=link
        self.maskSources,self.label=buildMaskSources(self.listCenter,self.listRadius,shape)
        self.spectraSources=[]
        self.spectraSourcesLmbda=spectraSourcesLmbda
        self.spectraSourcesWidth=spectraSourcesWidth
        self.maskAll=self.maskSources>0 
        
        #create "filament link" between point sources
        if link is not None:
            for k in link:
                self.linkGal(k[0],k[1],intens)
        
        for k in xrange(len(spectraSourcesLmbda)):
            self.spectraSources.append(createSpectra(spectraSourcesLmbda[k],shape[0],width=5))
            for i in xrange(shape[1]):
                for j in xrange(shape[2]):
                    if self.label[i,j]==k+1:
                        
                        if variation !=0:
                            spectra=createSpectra(spectraSourcesLmbda[k]+np.random.randint(variation),shape[0],width=5)
                        else:
                            spectra=self.spectraSources[k]
                        self.data[:,i,j]=spectra*self.maskSources[i,j]
        
        self.noise=noise
        self.rho=rho        
        
        if self.noise != None:
            self.dataClean=self.data.copy()
            if rho == 0: #no correlation
                self.data=self.data+np.random.normal(scale=self.noise,size=shape[0]*shape[1]*shape[2]).reshape(shape)
            else: #generate spatially correlated noise
                self.createCorrNoiseCube()
                self.data=self.data+self.noiseCube
            self.var=np.ones((shape))*self.noise**2
        else:
            self.dataClean=self.data
            self.var=np.ones((shape))

        cubeClean=Cube(data=self.dataClean,wcs=WCS(),wave=WaveCoord())
        cubeNoisy=Cube(data=self.data,var=self.var,wcs=WCS(),wave=WaveCoord())
        ra,dec=listCenter[0]
        self.src=Source.from_data(4000,ra+1,dec+1,origin='Simulated',cubes={'TRUTH_CUBE':cubeClean,'MUSE_CUBE':cubeNoisy})
        self.src.add_line(['LBDA_OBS','LINE'],[lmbda,"LYA"])
        
        #self.src.add_attr('SIMULATED_SOURCE',True,desc='Indicates to the detection method weither this is a real or semi-real source (with wcs for example) or a simulated one')
        
    def copy(self):
        cube=CubeSimMultiObj(self.shape,self.lmbda,self.noise,self.spectraSourcesLmbda,self.spectraSourcesWidth,self.listCenter,self.listRadius,self.link,self.intens)
        return cube
        
    def linkGal(self,obj1,obj2,intens):
        ### Attention
        ###!!!
        ### pour l'instant ne gere pas plusieurs link
        self.maskPoint=utils.line(self.listCenter[obj1],self.listCenter[obj2])

        dist=float((self.listCenter[obj1][0]-self.listCenter[obj2][0])**2+(self.listCenter[obj1][1]-self.listCenter[obj2][1])**2)
        listDist=[(self.listCenter[obj1][0]-k[0])**2+(self.listCenter[obj1][1]-k[1])**2 for k in self.maskPoint]
        listLmbda=[self.spectraSourcesLmbda[obj1]+(self.spectraSourcesLmbda[obj2]-self.spectraSourcesLmbda[obj1])*k/dist for k in listDist]
        listSpectra=[createSpectra(i,self.shape[0],width=5) for i in listLmbda]
        for i,k in enumerate(self.maskPoint):
            self.data[:,k[0],k[1]]=listSpectra[i]*intens 
        for k in self.maskPoint:
            self.maskAll[k]=True
        
    def createCorrNoiseCube(self):
        """
        Build covariance matrix : the idea is to have a correlation between two pixels decreasing with the distance
        so we first build a distance matrix then we ponderate by the rho coefficient and then we truncate 
        for low  correlation two avoid too many computations.
        Finally we generate as many of samples as there are wavelenght slices.
        """        
        self.noiseCube=np.zeros(self.shape)
        a=(np.indices((self.shape[1],self.shape[2]))).reshape(2,self.shape[1]*self.shape[2]).T
        dist=dists.cdist(a,a)
        self.covar= self.noise*self.rho**dist
        self.covar[self.covar<0.01]=0
        for k in xrange(self.shape[0]):
            self.noiseCube[k]=np.random.multivariate_normal(np.zeros(self.shape[1]*self.shape[2]),self.covar,size=1).reshape(self.shape[1],self.shape[2])
                
        
def buildHaloSpectra(lmbda,width,size):
    haloSpectra=np.zeros(size)
    haloSpectra[lmbda-width/2.:lmbda+width/2.]=sst.norm.pdf(np.arange(-width/2.,width/2.),scale=2)
    haloSpectra=haloSpectra*1/np.max(haloSpectra)
    return haloSpectra        

    
def createSpectra(pos,size,width=5):
    size=np.floor(size/2)*2 ## assure size is odd
    xvar=np.linspace(0,size,size+1)  
    spectrum = sst.norm.pdf(xvar,loc=pos,scale=width/2.35)
    spectrum=spectrum/np.max(spectrum)
    return spectrum

def buildMaskSources(listCenter,listRadius,shape):
    x,y=np.mgrid[0:shape[1], 0:shape[2]]
    #on crée un profil spatial gaussien
    mask=np.zeros((shape[1],shape[2]))
    label=np.zeros((shape[1],shape[2]))  
    for k in xrange(len(listCenter)):
        zGalaxy = multivariate_normal.pdf(np.swapaxes(np.swapaxes([x,y],0,2),0,1),mean=listCenter[k], cov=[[listRadius[k]/1.17, 0], [0, listRadius[k]/1.17]])
        zGalaxy= zGalaxy*1/np.max(zGalaxy)
        zGalaxy[zGalaxy<0.1]=0
        label[zGalaxy>0]=k+1
        mask=mask+zGalaxy    
    #On tronque le profil gaussien d et de la galaxie à 0.1
    
    return mask,label
        



def createSemiRealSource(srcData,cube,SNR):
    """
    Create a Source object with a cube of data mixing real noise from a MUSE (sub-)cube and signal
    of a simulated halo object. The object is centered in the real MUSE subcube 
    
    """
    dec,ra = cube.wcs.pix2sky([cube.shape[1]/2,cube.shape[2]/2])[0]
    lmbda = cube.wave.coord(cube.shape[0]/2)
    cube.data=cube.data+SNR*srcData
    src=Source.from_data(4000,ra,dec,origin='Simulated',cubes={'MUSE_CUBE':cube})
    src.add_line(['LBDA_OBS','LINE'],[lmbda,"LYA"])
    src.images['TRUTH_DET_BIN_ALL']= Image(data=obj.maskSources>0 )
    return src