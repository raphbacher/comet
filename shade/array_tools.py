# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:35:12 2015

@author: raphael
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def normArr(arrIn, varIn=None):
    res = np.empty_like(arrIn)
    resVar = np.empty_like(arrIn)
    if len(arrIn.shape) == 3:
        for i in range(res.shape[1]):
            for j in range(res.shape[2]):
                res[:, i, j] = arrIn[:, i, j]/np.sqrt(np.sum(arrIn[:, i, j]**2))
                if varIn is not None:
                    resVar[:, i, j] = varIn[:, i, j] / np.sum(arrIn[:, i, j]**2)
    elif len(arrIn.shape) == 2:
        for i in range(res.shape[1]):
            res[:, i] = arrIn[:, i] / np.sqrt(np.sum(arrIn[:, i]**2))
            if varIn is not None:
                resVar[:, i] = varIn[:, i] / np.sum(arrIn[:, i]**2)
    if varIn is not None:
        return res, resVar
    return res
