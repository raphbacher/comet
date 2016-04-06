# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:27:22 2015

@author: raphael
"""

import scipy.stats as sst
from scipy.stats import norm
import numpy as np
import multiprocessing
from scipy.optimize import minimize
import scipy.signal as ssl
from mpdaf.obj import Image



def fine_clipping(Image1, niter = 3, fact_value = 0.9, Pmin = 0, Pmax = -1, Qmin = 0, Qmax = -1):
    P1,Q1 = Image1.shape
    if Qmax == -1:
        Qmax = Q1
    if Pmax == -1:
        Pmax = P1
    Image = Image1[Pmin:Pmax, Qmin:Qmax].copy()
    P,Q = Image.shape
    Quartile1 = np.percentile(Image.data, 25)
    Quartile3 = np.percentile(Image.data, 75)
    IQR= Quartile3 - Quartile1
    med = np.median(Image.data)
    sigestQuant = IQR/1.349
    x = np.reshape(Image.data, P*Q)
    
    xclip = x
    
    facttrunc = norm.ppf(fact_value)
    correction = norm.ppf((0.75*( 2*norm.cdf(facttrunc)-1 ) + (1 - norm.cdf(facttrunc)) )) - norm.ppf(0.25*( 2*norm.cdf(facttrunc)-1 ) + (1 - norm.cdf(facttrunc)) )
    medclip = np.nanmedian(xclip)
    x=x.filled(medclip)
    xclip=xclip.filled(medclip)
    qlclip = np.percentile(xclip, 25)
    stdclip = 2.*(medclip - qlclip)/1.349    
    oldmedclip=1.
    
    for i in xrange(niter):
        try:
            xclip = x[ np.where( ((x-medclip) < facttrunc*stdclip) &  ( (x-medclip) > -facttrunc*stdclip )  ) ] # on garde la symetrie dans la troncature
            medclip = np.median(xclip)
            qlclip = np.percentile(xclip, 25)
            stdclip = 2*(medclip - qlclip)/correction

        except:
            print "error normalizing"
            return Image1
                
    xclip2 = x[np.where( ((x-medclip) <0) & ((x-medclip) > -3*stdclip)) ]
    correctionTrunc= np.sqrt( 1. +(-3.*2.* norm.pdf(3.)) / (2.*norm.cdf(3.) -1.) )
    stdclip2 = np.sqrt( np.mean( (xclip2-medclip)**2)) / correctionTrunc
    
    
    Image1.data = (Image1.data- medclip)/stdclip2
    return Image1


def recenter(Image1, niter = 3, lmbda=1., fact_value = 0.8, Pmin = 0, Pmax = -1, Qmin = 0, Qmax = -1):
    P1,Q1 = Image1.shape
    if Qmax == -1:
        Qmax = Q1
    if Pmax == -1:
        Pmax = P1
    Image = Image1[Pmin:Pmax, Qmin:Qmax].copy()
    P,Q = Image.shape
    Quartile1 = np.percentile(Image.data, 25)
    Quartile2 = np.percentile(Image.data, 50)
    IQR= Quartile2 - Quartile1
    med = np.median(Image.data)
    stdclip = IQR*1.48
    x = np.reshape(Image.data, P*Q)
    medclip=0.
    
    oldmedclip=1.
    for i in xrange(niter):
        if medclip!=oldmedclip:
            
            xclip = x[ np.where( ((x-medclip) < lmbda*stdclip) &  ( (x-medclip) > -lmbda*stdclip )  ) ] # on garde la symetrie dans la troncature
            oldmedclip=medclip
            medclip = np.mean(xclip)
            qlclip = np.percentile(xclip, 25)
            stdclip = (medclip - qlclip)*1.48
        else:
            break
    
    
    Image1.data = Image1.data- medclip
    return Image1
    

def recenterMul(cube,w=None):
    pool=multiprocessing.Pool()
    
    try:
        print 'starting the pool map'
        res=np.array(pool.map(recenter,[i for i in cube]))
        print res.shape
        pool.close()
        print 'pool map complete'
        if w is None:
            res=res.reshape(cube.shape[0],cube.shape[1],cube.shape[2])
        else:
            res=res.reshape(2*w,cube.shape[1],cube.shape[2])
    except KeyboardInterrupt:
        print 'got ^C while pool mapping, terminating the pool'
        pool.terminate()
        print 'pool is terminated'    
    except Exception, e:
        print 'got exception: %r, terminating the pool' % (e,)
        pool.terminate()
        print 'pool is terminated'
    
    finally:
        pool.join()
    return res
    

def getParamNoise(Image1, niter = 10, fact_value = 0.9, Pmin = 0, Pmax = -1, Qmin = 0, Qmax = -1):
    P1,Q1 = Image1.shape
    if Qmax == -1:
        Qmax = Q1
    if Pmax == -1:
        Pmax = P1
    Image = Image1[Pmin:Pmax, Qmin:Qmax].copy()
    P,Q = Image.shape
    Quartile1 = np.percentile(Image, 25)
    Quartile3 = np.percentile(Image, 75)
    IQR= Quartile3 - Quartile1
    med = np.median(Image)
    sigestQuant = IQR/1.349
    x = np.reshape(Image, P*Q)
    xclip = x
    
    facttrunc = norm.ppf(fact_value)
    correction = norm.ppf((0.75*( 2*norm.cdf(facttrunc)-1 ) + (1 - norm.cdf(facttrunc)) )) - norm.ppf(0.25*( 2*norm.cdf(facttrunc)-1 ) + (1 - norm.cdf(facttrunc)) )
    medclip = np.median(xclip)
    qlclip = np.percentile(xclip, 25)
    stdclip = 2.*(medclip - qlclip)/1.349
        
    for i in xrange(niter):
        xclip = x[ np.where( ((x-medclip) < facttrunc*stdclip) &  ( (x-medclip) > -facttrunc*stdclip )  ) ] # on garde la symetrie dans la troncature
        if len(xclip)==0:
            break
        medclip = np.median(xclip)
        qlclip = np.percentile(xclip, 25)
        stdclip = 2*(medclip - qlclip)/correction
        
    xclip2 = x[np.where( ((x-medclip) <0) & ((x-medclip) > -3*stdclip)) ]
    correctionTrunc= np.sqrt( 1. +(-3.*2.* norm.pdf(3.)) / (2.*norm.cdf(3.) -1.) )
    stdclip2 = np.sqrt( np.mean( (xclip2-medclip)**2)) / correctionTrunc
    
    
    return medclip,stdclip2
    
def getParamNoiseMul(cube):
    pool=multiprocessing.Pool()
    
    try:
        print 'starting the pool map'
        res=np.array(pool.map(getParamNoise,[i for i in cube]))
        pool.close()
        print 'pool map complete'

    except KeyboardInterrupt:
        print 'got ^C while pool mapping, terminating the pool'
        pool.terminate()
        print 'pool is terminated'    
    
    except Exception, e:
        print 'got exception: %r, terminating the pool' % (e,)
        pool.terminate()
        print 'pool is terminated'
    
    finally:
        pool.join()
    return res
    
    
def getStudentParam(Image1, niter = 10, fact_value = 0.9, Pmin = 0, Pmax = -1, Qmin = 0, Qmax = -1, runLikelihood=False):
    P1,Q1 = Image1.shape
    if Qmax == -1:
        Qmax = Q1
    if Pmax == -1:
        Pmax = P1
    Image = Image1[Pmin:Pmax, Qmin:Qmax].copy()
    P,Q = Image.shape
    medclip,sigclip=getParamNoise(Image1, niter, fact_value, Pmin, Pmax, Qmin, Qmax)
    kurto=sst.moment(Image1,4,None)/sst.moment(Image1,2,None)**2-3
    mu=(4*kurto+6)/kurto
    sigmaEst=np.sum((Image1[Image1<medclip]-medclip)**2)/len(Image1[Image1<medclip])
    sEst=np.sqrt((mu-2)/mu*sigmaEst)
    
    if runLikelihood==True:
        initParam=(mu,sEst)  
        results=minimize(logLH,initParam,method='Powell',args=(Image[Image<medclip],medclip))
        mu,sEst=results.x
        
    return medclip,sEst,mu,sigclip
    
def logLH(params,x,m):
    mu=params[0]
    s=params[1]
    return -np.sum(sst.t.logpdf(x,loc=m,scale=s,df=mu))
        
def getStudentParamMul(cube):
    pool=multiprocessing.Pool()
    
    try:
        print 'starting the pool map'
        res=np.array(pool.map(getStudentParam,[i for i in cube]))
        print res.shape
        pool.close()
        print 'pool map complete'

    except KeyboardInterrupt:
        print 'got ^C while pool mapping, terminating the pool'
        pool.terminate()
        print 'pool is terminated'    
    
    except Exception, e:
        print 'got exception: %r, terminating the pool' % (e,)
        pool.terminate()
        print 'pool is terminated'
    
    finally:
        pool.join()
    return res
    
def Image_conv( im, tab):
    """ Defines the convolution between an Image object and an array. Designed to be used with the multiprocessing function 'FSF_convolution_multiprocessing'
        
        :param im: Image object 
        :type im: class 'mpdaf.obj.Image'
        :param tab: array containing the convolution kernel
        :type tab: array
        :return: array
        :rtype: array
        
    """
    res = ssl.convolve(im.data, tab, 'full')
    a,b = tab.shape
    im_tmp = Image(data = res[int(a-1)/2:im.data.shape[0] + (a-1)/2 ,(b-1)/2:im.data.shape[1]+(b-1)/2 ])
    return im_tmp.data