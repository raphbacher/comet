#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:52:55 2017

@author: raphael.bacher@gipsa-lab.fr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np


def Moffat(r, alpha, beta):
    """
    Compute Moffat values for array of distances *r* and Moffat parameters *alpha* and *beta*
    """
    return (beta-1)/(math.pi*alpha**2)*(1+(r/alpha)**2)**(-beta)

def generateMoffatIm(center=(12,12),shape=(25,25),alpha=2,beta=2.5,dx=0.,dy=0.,dim='MUSE'):
    """
    By default alpha is supposed to be given in arsec, if not it is given in MUSE pixel.
    a,b allow to decenter slightly the Moffat image.
    """
    ind = np.indices(shape)
    r = np.sqrt(((ind[0]-center[0]+dx)**2 + ((ind[1]-center[1]+dy))**2))
    if dim == 'MUSE':
        r = r*0.2
    elif dim == 'HST':
        r = r*0.03
    res = Moffat(r, alpha, beta)
    res = res/np.sum(res)
    return res
