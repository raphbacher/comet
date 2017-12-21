# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:27:22 2015

@author: raphael
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.stats as sst
from scipy.stats import norm
import numpy as np
import multiprocessing
from scipy.optimize import minimize
import scipy.signal as ssl
from mpdaf.obj import Image


def fine_clipping(Image, niter=20, fact_value=0.9, Pmin=0, Pmax=-1, Qmin=0,
                  Qmax=-1, unmask=True):
    P, Q = Image.shape
    if Qmax == -1:
        Qmax = Q
    if Pmax == -1:
        Pmax = P
    Image1 = Image[Pmin:Pmax, Qmin:Qmax].copy()
    P1, Q1 = Image1.shape
    Quartile1 = np.percentile(Image1.data, 25)
    Quartile3 = np.percentile(Image1.data, 75)
    IQR = Quartile3 - Quartile1
    med = np.median(Image1.data)
    sigestQuant = IQR/1.349
    if unmask is True:
        x = np.reshape(Image1.data.data, P1*Q1)
    else:
        x = np.reshape(Image1.data, P1*Q1)
    xclip = x

    facttrunc = norm.ppf(fact_value)
    correction = norm.ppf((0.75*(2*norm.cdf(facttrunc)-1) +
                           (1 - norm.cdf(facttrunc))))\
                           - norm.ppf(0.25*(2*norm.cdf(facttrunc)-1)
                           + (1 - norm.cdf(facttrunc)))
    medclip = np.nanmedian(xclip)
# necessary if nan values
#    x=x.filled(medclip)
#    xclip=xclip.filled(medclip)
#    xclip[np.isnan(xclip)]=medclip
#    x[np.isnan(x)]=medclip
    qlclip = np.percentile(xclip, 25)
    stdclip = 2.*(medclip - qlclip)/1.349
    oldmedclip = 1.

    for i in xrange(niter):
        try:
            # on garde la symetrie dans la troncature
            xclip = x[np.where(((x-medclip) < facttrunc*stdclip) &
                               ((x-medclip) > -facttrunc*stdclip))]
            medclip = np.median(xclip)
            qlclip = np.percentile(xclip, 25)
            stdclip = 2*(medclip - qlclip)/correction

        except:
            print("error normalizing")
            return Image1

    xclip2 = x[np.where(((x-medclip) < 0) & ((x-medclip) > -3*stdclip))]
    correctionTrunc= np.sqrt(1. + (-3.*2.* norm.pdf(3.)) / (2.*norm.cdf(3.) -1.))
    stdclip2 = np.sqrt(np.mean((xclip2-medclip)**2)) / correctionTrunc

    Image1.data = (Image1.data - medclip)/stdclip2
    return Image1


def fine_clipping2(Image, niter=10, fact_value=0.9, Pmin=0, Pmax=-1,
                   Qmin=0, Qmax=-1, unmask=True):
    P, Q = Image.shape
    if Qmax == -1:
        Qmax = Q
    if Pmax == -1:
        Pmax = P
    Image1 = Image[Pmin:Pmax, Qmin:Qmax].copy()
    P1, Q1 = Image1.shape
    if unmask is True:
        x = np.reshape(Image1.data.data, P1*Q1)
    else:
        x = np.reshape(Image1.data, P1*Q1)
    x_sorted = np.sort(x)
    Quartile1 = percent(x_sorted, 25)
    Quartile3 = percent(x_sorted, 75)
    IQR = Quartile3 - Quartile1
    med = middle(x_sorted)
    fact_IQR = norm.ppf(0.75) - norm.ppf(0.25)
    sigestQuant = IQR/fact_IQR

    xclip = x_sorted

    facttrunc = norm.ppf(fact_value)
    cdf_facttrunc = norm.cdf(facttrunc)
    correction = norm.ppf((0.75*( 2*cdf_facttrunc-1 ) + (1 - cdf_facttrunc) )) - norm.ppf(0.25*( 2*cdf_facttrunc-1 ) + (1 - cdf_facttrunc) )
    medclip = middle(xclip)
    qlclip = percent(xclip, 25)
    stdclip = 2.*(medclip - qlclip) / fact_IQR
    oldmedclip = 1.
    oldstdclip = 1.

    i = 0
    while (oldmedclip != medclip) and (i < niter):
        lim=np.searchsorted(x_sorted,[medclip-facttrunc*stdclip,medclip+facttrunc*stdclip])
        xclip = x_sorted[lim[0]:lim[1]]
        oldoldmedclip = oldmedclip

        oldmedclip = medclip
        oldstdclip = stdclip

        medclip = middle(xclip)

        qlclip = percent(xclip, 25)
        stdclip = 2*np.abs(medclip - qlclip)/correction

        if oldoldmedclip == medclip:  # gestion des cycles

            if stdclip > oldstdclip:
                break
            else:
                stdclip = oldstdclip
                medclip = oldmedclip
        i += 1

    xclip2 = x_sorted[np.where(((x_sorted-medclip) < 0) &
                               ((x_sorted-medclip) > -3*stdclip))]
    correctionTrunc = np.sqrt(1. + (-3. * 2. * norm.pdf(3.)) /
                              (2. * norm.cdf(3.) - 1.))
    stdclip2 = np.sqrt(np.mean((xclip2-medclip)**2)) / correctionTrunc

    Image1.data = (Image1.data - medclip)/stdclip2

    return Image1


def middle(L):

    n = int(len(L))
    return (L[n//2] + L[n//2-1]) / 2.0


def percent(L, q):
    """L np.array, q betwwen 0-100"""
    n0 = q/100. * len(L)
    n = int(np.floor(n0))
    if n >= len(L):
        return L[-1]
    if n >= 1:
        if n == n0:
            return L[n-1]
        else:
            return (L[n-1]+L[n])/2.0
    else:
        return L[0]


def recenter(Image1, niter=3, lmbda=1., fact_value=0.8, Pmin=0, Pmax=-1,
             Qmin=0, Qmax=-1):
    P1, Q1 = Image1.shape
    if Qmax == -1:
        Qmax = Q1
    if Pmax == -1:
        Pmax = P1
    Image = Image1[Pmin:Pmax, Qmin:Qmax].copy()
    P, Q = Image.shape
    Quartile1 = np.percentile(Image.data, 25)
    Quartile2 = np.percentile(Image.data, 50)
    IQR = Quartile2 - Quartile1
    med = np.median(Image.data)
    stdclip = IQR*1.48
    x = np.reshape(Image.data, P*Q)
    medclip = 0.

    oldmedclip = 1.
    for i in xrange(niter):
        if medclip != oldmedclip:
            # on garde la symetrie dans la troncature
            xclip = x[np.where(((x-medclip) < lmbda*stdclip) &
                               ((x-medclip) > -lmbda*stdclip))]
            oldmedclip = medclip
            medclip = np.mean(xclip)
            qlclip = np.percentile(xclip, 25)
            stdclip = (medclip - qlclip)*1.48
        else:
            break

    Image1.data = Image1.data - medclip
    return Image1


def recenterMul(cube, w=None):
    pool = multiprocessing.Pool()

    try:
        print('starting the pool map')
        res = np.array(pool.map(recenter, [i for i in cube]))
        print(res.shape)
        pool.close()
        print('pool map complete')
        if w is None:
            res = res.reshape(cube.shape[0], cube.shape[1], cube.shape[2])
        else:
            res = res.reshape(2*w, cube.shape[1], cube.shape[2])
    except KeyboardInterrupt:
        print('got ^C while pool mapping, terminating the pool')
        pool.terminate()
        print('pool is terminated')
    except Exception as e:
        print('got exception: %r, terminating the pool' % (e,))
        pool.terminate()
        print('pool is terminated')

    finally:
        pool.join()
    return res


def getParamNoise(Image1, niter=10, fact_value=0.9, Pmin=0, Pmax=-1,
                  Qmin=0, Qmax=-1):
    P1, Q1 = Image1.shape
    if Qmax == -1:
        Qmax = Q1
    if Pmax == -1:
        Pmax = P1
    Image = Image1[Pmin:Pmax, Qmin:Qmax].copy()
    P, Q = Image.shape
    Quartile1 = np.percentile(Image, 25)
    Quartile3 = np.percentile(Image, 75)
    IQR = Quartile3 - Quartile1
    med = np.median(Image)
    sigestQuant = IQR/1.349
    x = np.reshape(Image, P*Q)
    xclip = x

    facttrunc = norm.ppf(fact_value)
    correction = norm.ppf((0.75*(2*norm.cdf(facttrunc)-1)
                           + (1 - norm.cdf(facttrunc)))) \
        - norm.ppf(0.25*(2*norm.cdf(facttrunc)-1) + (1 - norm.cdf(facttrunc)))

    medclip = np.median(xclip)
    qlclip = np.percentile(xclip, 25)
    stdclip = 2.*(medclip - qlclip)/1.349

    for i in xrange(niter):
        # on garde la symetrie dans la troncature
        xclip = x[np.where(((x-medclip) < facttrunc*stdclip) &
                           ((x-medclip) > -facttrunc*stdclip))]

        if len(xclip) == 0:
            break
        medclip = np.median(xclip)
        qlclip = np.percentile(xclip, 25)
        stdclip = 2*(medclip - qlclip)/correction

    xclip2 = x[np.where(((x-medclip) < 0) & ((x-medclip) > -3*stdclip))]
    correctionTrunc = np.sqrt(1. + (-3. * 2. * norm.pdf(3.)) /
                              (2. * norm.cdf(3.) - 1.))
    stdclip2 = np.sqrt(np.mean((xclip2-medclip)**2)) / correctionTrunc

    return medclip, stdclip2


def getParamNoiseMul(cube):
    pool = multiprocessing.Pool()

    try:
        print('starting the pool map')
        res = np.array(pool.map(getParamNoise, [i for i in cube]))
        pool.close()
        print('pool map complete')

    except KeyboardInterrupt:
        print('got ^C while pool mapping, terminating the pool')
        pool.terminate()
        print('pool is terminated')

    except Exception as e:
        print('got exception: %r, terminating the pool' % (e,))
        pool.terminate()
        print('pool is terminated')

    finally:
        pool.join()
    return res


def getStudentParam(Image1, niter=10, fact_value=0.9, Pmin=0, Pmax=-1, Qmin=0,
                    Qmax=-1, runLikelihood=False):
    P1, Q1 = Image1.shape
    if Qmax == -1:
        Qmax = Q1
    if Pmax == -1:
        Pmax = P1
    Image = Image1[Pmin:Pmax, Qmin:Qmax].copy()
    P, Q = Image.shape
    medclip, sigclip = getParamNoise(Image1, niter, fact_value, Pmin,
                                     Pmax, Qmin, Qmax)
    kurto = sst.moment(Image1, 4, None) / sst.moment(Image1, 2, None)**2 - 3
    mu = (4*kurto+6)/kurto
    sigmaEst = np.sum((Image1[Image1 < medclip]-medclip)**2)\
        / len(Image1[Image1 < medclip])
    sEst = np.sqrt((mu-2)/mu*sigmaEst)

    if runLikelihood is True:
        initParam = (mu, sEst)
        results = minimize(logLH, initParam, method='Powell',
                           args=(Image[Image < medclip], medclip))
        mu, sEst = results.x

    return medclip, sEst, mu, sigclip


def logLH(params, x, m):
    mu = params[0]
    s = params[1]
    return -np.sum(sst.t.logpdf(x, loc=m, scale=s, df=mu))


def getStudentParamMul(cube):
    pool = multiprocessing.Pool()

    try:
        print('starting the pool map')
        res = np.array(pool.map(getStudentParam, [i for i in cube]))
        print(res.shape)
        pool.close()
        print('pool map complete')

    except KeyboardInterrupt:
        print('got ^C while pool mapping, terminating the pool')
        pool.terminate()
        print('pool is terminated')

    except Exception as e:
        print('got exception: %r, terminating the pool' % (e,))
        pool.terminate()
        print('pool is terminated')

    finally:
        pool.join()
    return res


def Image_conv(im, tab, unmask=True):
    """ Defines the convolution between an Image object and an array.
    Designed to be used with the multiprocessing function
    'FSF_convolution_multiprocessing'.

        :param im: Image object
        :type im: class 'mpdaf.obj.Image'
        :param tab: array containing the convolution kernel
        :type tab: array
        :param unmask: if True use .data of masked array (faster computation)
        :type unmask: bool
        :return: array
        :rtype: array

    """
    if unmask is True:
        res = ssl.fftconvolve(im.data.data, tab, 'full')
    else:
        res = ssl.fftconvolve(im.data, tab, 'full')
    a, b = tab.shape
    im_tmp = Image(data=res[int(a-1)//2:im.data.shape[0] + (a-1)//2,
                            (b-1)//2:im.data.shape[1]+(b-1)//2])
    return im_tmp.data
