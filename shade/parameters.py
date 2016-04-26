# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 03:29:26 2015

@author: raphael
"""

import cPickle as pickle
import os

DEFAULT_FSF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           'fsf_HDFS_v1-24.pk')

class Params():
    
    def __init__(self,
                 LW=20, 
                 SW=None,
                 LBDA=3641,
                 sim=False,
                 lmbdaShift=7,
                 version='V1.0',
                 fsf=DEFAULT_FSF,
                 ):
        """
        Param: int *LW*, Lambda Window where the correlation test will occur (that must cover the half-width of the line emission)
        Param: int *SW*, Spatial Window for the exploration, if None the cube is fully explored spatially
        Param: bool *sim*, indicates if given sources are simulated ones (without wcs and wave objects)
        Param: int *lmbdaShift*, maximum shift in one direction to construct a family of target spectra
        from the estimated source spectrum. A dictionary with 2*lmbdaShift+1 spectra will be built.
        Param: string *centering*, choose to center all spectra ('all') or 'none' or only the target spectra ('ref')
        Param: bool *norm*, choose to norm (correlation approach) or not (matched filter approach)
        Param: string *fsf*, path to an fsf file (3D array)
        """

        self.LW=LW
        self.SW=SW
        self.sim=sim
        self.lmbdaShift=lmbdaShift
        self.version=version
        self.origin=['SHADE',version]
        self.fsf=pickle.load(open(fsf,'rb'))[0:LBDA,:,:]
        
class ParamsPreProcess():
    """
    Param: bool *allCube*, wheither to process the whole cube at once then reform the sources
    or process each source datacube independently
    Param: string methodRC, choice of the method for remove the continuum 
    (for now only median filter 'medfilt', lts is on his way)
    Param: int windowRC, window for median filter (in this case it is the whole window not the half-size)
    Param: int Pmin,Pmax,Qmin,Qmax, trim some borders of the datacube to avoid some problems, used only
    with the allCube processing. 0 for Pmin and -1 for Pmax mean no trimming.
    Param: int lmbdaMin and lmbdaMax, trim some wavelength if allCube processing.
    Param: bool forceProcess, force a new preprocessing of sources even if sources have already a "PROCESS_CUBE"
    Param: unmask=True, unmask masked array with median filled values to speed up calculations. MUST BE SET TO TRUE FOR NOW (unsolved bugs due to nan values)
    Param: shiftLamdaDetectin=0, to test false detection in a spectrally shifted (empty) area
    """
    def __init__(self,
                 allCube=False,
                 methodRC='medfilt',
                 windowRC=101,
                 Pmin=0,
                 Pmax=-1,
                 Qmin=0,
                 Qmax=-1,
                 lmbdaMin=0,
                 lmbdaMax=None,
                 forceProcess=False,
                 unmask=True,
                 shiftLambdaDetection=0
                 ):
        self.allCube=allCube
        self.methodRC=methodRC
        self.windowRC=windowRC
        self.Pmin=Pmin
        self.Pmax=Pmax
        self.Qmin=Qmin
        self.Qmax=Qmax
        self.lmbdaMin=lmbdaMin
        self.lmbdaMax=lmbdaMax
        self.forceProcess=forceProcess
        self.unmask=unmask
        self.shiftLambdaDetection=shiftLambdaDetection
        
    
class ParamsDetection():
    """
    Param: int *windowRef*, spatial window (half-width) for computing the reference spectrum (by averaging)
    at the center of the galaxy.
    Param: string *centering*, center (with 'all') or not (with 'none') the spectra to be tested or 
    only the reference spectra (with 'ref')
    Param: bool *norm*, normalize spectra or not in the correlation test.
    Param: list *listRef*, proposed dictionnary of spectra (None by default as it is learned on data )
    """


    def __init__(self,
                 windowRef=1,
                 centering='none',
                 norm=True,
                 listRef=None
                 ):
        self.windowRef=windowRef
        self.centering=centering
        self.norm=norm
        self.listRef=listRef



class ParamsPostProcess():
    """
    Param: int *threshold*
    Param: bool *FDR*, apply threshold in FDR instead of PFA
    Param: bool *qvalue*, compute or not q-values (that are more or less FDR-pvalues)
    Param: bool *newSource*, save results in new sources objects instead of current sources
    Param: bool *resizeCube*, resize MUSE_CUBE in sources accordingly with PROCESS_CUBE
    """
    def __init__(self,
                 threshold=0.1,
                 FDR=True,
                 qvalue=True,
                 newSource=True,
                 resizeCube=True,
                 ):
        self.FDR=FDR
        self.threshold=threshold
        self.thresholdFDR=0
        self.qvalue=qvalue
        self.resizeCube=resizeCube
        self.newSource=newSource
