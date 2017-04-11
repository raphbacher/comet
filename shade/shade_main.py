# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 03:54:02 2015

@author: raphael
"""

from mpdaf.sdetect.source import Source
from mpdaf.sdetect import Catalog
from astropy.io import fits as pyfits
from mpdaf.obj import Cube
import parameters
import preprocessing as prep
import postprocessing as postp
import detection


class SHADE():


    def __init__(self,cube=None, processedCube=None,catalog=None, listSources=None, listID = None,params=None):
        """
        Several Input choices are available:
        * List of MUSE Sources : These sources needs a MUSE_CUBE in their cubes extension.
        * Complete cube with a MUSE catalogue -> all Lyman-alpha emitters will be treated
        * Complete cube, catalogue and listID -> all the source of the ID list will be treated
        An already preprocessed processedCube can also by passed along.
        Param: Cube object *cube*, MUSE datacube (optionnal if listSources is defined)
        Param: Cube object *processedCube*, preprocessed datacube
        Param: String *catalog*, filename of a MUSE catalog
        Param: list of Sources object *listSources* (optionnal)
        Param: list of sources IDs *listID*, list of sources to extract from the cube using the catalog.
        Param: objet Params *params*, parameters for the method, if not defined,
        default parameters are used

        """
        if params is None:
            self.params=parameters.Params()
        else:
            self.params=params
        if cube is not None:
            self.cube=cube
        else:
            self.cube=None
        if catalog is not None:
            self.catalog=Catalog.read(catalog)
        if processedCube is not None:
            self.processedCube=processedCube
        else:
            self.processedCube=None

        if listSources is not None:
            self.listSources=listSources

        elif listSources is None and listID is None:
            hdulist=pyfits.open(catalog)
            listID=[]
            for k in xrange(len(hdulist[1].data)):
                if hdulist[1].data[k][1]=='Lya' and hdulist[1].data[k][4]>0:#We get all Lyman alpha with a defined redshift
                    listID.append(hdulist[1].data[k][0])
            self.listSources=[]
            for k in listID:
                self.listSources.append(self.sourceFromCatalog(k))

        elif listID is not None:
            self.listSources=[]
            for k in listID:
                self.listSources.append(self.sourceFromCatalog(k))

        self.listCorrArr=[]
        self.listPvalMap=[]
        self.preprocessing=None
        self.postprocessing=None
        self.paramsPreProcess=parameters.ParamsPreProcess()
        self.paramsPostProcess=parameters.ParamsPostProcess()
        self.paramsDetection=parameters.ParamsDetection()


    def preprocess(self,paramsPreProcess=None):
        """
        Preprocess the sources and store processed cube in a PROCESS_CUBE cube. If a source has already a PROCESS_CUBE, it will not be processed again.
        """
        if paramsPreProcess is not None:
            self.paramsPreProcess=paramsPreProcess

        if self.preprocessing is None:
            self.preprocessing=prep.Preprocessing(cube=self.cube,listSources=self.listSources,processedCube=self.processedCube,params=self.params,paramsPreProcess=self.paramsPreProcess)

        #In some cases (lot of sources with some overlapping areas) it can be interesting to process all the cube
        #and then to extract processed data for the sources instead of processing several times the same data

        if self.paramsPreProcess.allCube == True:
            self.preprocessing.processSrcWithCube()
        else:
            self.preprocessing.processSrc()



    def detect(self,paramsDetection=None):
        """
        Compute the detection test. At this point sources must contain a PROCESS_CUBE
        of shape [LW-lmbda:LW+lmbda,center[0]-SW:center[0]+SW,center[1]-SW:center[1]+SW]
        or just [LW-lmbda:LW+lmbda,:,:] if SW is None.
        """
        if paramsDetection is not None:
            self.paramsDetection=paramsDetection

        self.detection=detection.Detection(listSources=self.listSources,params=self.params,paramsPreProcess=self.paramsPreProcess,paramsDetection=self.paramsDetection)

        self.listPvalMap,self.listIndexMap=self.detection.detect()
        self.listCorrArr=self.detection.listCorrArr

    def postprocess(self,rawCube=None,paramsPostProcess=None):
        if paramsPostProcess is not None:
            self.paramsPostProcess=paramsPostProcess

        if rawCube is not None:
            cube=Cube(rawCube)
        else:
            cube=self.cube
        if self.postprocessing is None:
            self.postprocessing=postp.Postprocess(cube,self.listSources,self.listPvalMap,self.listIndexMap,params=self.params,paramsPreProcess=self.paramsPreProcess,paramsDetection=self.paramsDetection,paramsPostProcess=self.paramsPostProcess,)
        self.postprocessing.paramsPostProcess=self.paramsPostProcess
        self.postprocessing.createResultSources()
        if self.paramsPostProcess.newSource==True:
            self.listResultSources=self.postprocessing.listResultSources



    def sourceFromCatalog(self,ID):
        """
        Build source object from ID and catalog. By default, MUSE_CUBE in each source are not resized.
        """
        try:
            ra=self.catalog[self.catalog['ID']==ID]['RA'][0]
            dec=self.catalog[self.catalog['ID']==ID]['DEC'][0]
            z=self.catalog[self.catalog['ID']==ID]['Z_MUSE'][0]
        except:
            ra=self.catalog[self.catalog['ID']==ID]['Ra'][0]
            dec=self.catalog[self.catalog['ID']==ID]['Dec'][0]
            z=self.catalog[self.catalog['ID']==ID]['Z'][0]

        cubeData=self.cube
        src=Source.from_data(ID, ra, dec, origin=['SHADE Intern Format','1.0',self.cube.filename,'1.0'],cubes={'MUSE_CUBE':cubeData})
        src.add_line(['LBDA_OBS','LINE'],[(z+1)*1215.668,"LYALPHA"])
        return src