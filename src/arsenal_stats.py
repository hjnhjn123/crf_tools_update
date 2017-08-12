# -*- coding: utf-8 -*-
from __future__ import (unicode_literals, print_function, division)

import pandas as pd
import numpy as np


##########################################################################################

##########################################################################################


## HDF5 Processing

def df2set(df, title=False):
    return {i for j in df.as_matrix() for i in j} if title == False else \
        {i.title() for j in df.as_matrix() for i in j}


def df2list(df):
    return [i for j in df.as_matrix() for i in j]


def df2dic(df):
    """
    use pd.DataFrame.iloc to extract specific columns or rows
    :param df:
    :return:
    """
    return {k: v for (k, v) in zip(df.iloc[:, 0], df.iloc[:, 1])}

def df2hdf(out_hdf, hdf_keys, mode, *dfs):
    """
    Store single or multiple dfs to one hdf5 file
    :param dfs: single of multiple dfs
    :param out_hdf: the output file
    :param hdf_keys: [key for hdf]
    """
    for j, k in zip(dfs, hdf_keys):
        j.to_hdf(out_hdf, k, table=True, mode=mode)


def hdf2df(in_hdf, hdf_keys):
    """
    Read a hdf5 file and return all dfs
    :param in_hdf: a hdf5 file 
    :param hdf_keys: 
    :return a dict of df
    """
    return {i: pd.read_hdf(in_hdf, i) for i in hdf_keys}


##########################################################################################
def map_dic2df(df, col_name, feature_dict):
    df[col_name] = df.iloc[:, 0].map(feature_dict)
    return df.replace(np.nan, '0')
