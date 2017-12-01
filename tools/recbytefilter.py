#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 16:42:16 2017

filters record length bytes form binary file

@author: Theo Olsthoorn
"""
import numpy as np

def filter_recbytes(fin, fout=None, vartype=np.int32):
    '''copies the containts of binary file fin to output file with record-length bytes removed
    parameters
    ----------
    fin : str
        name of binary input file
    out: str
        name of binary output file, default is fname + '_out' + ext
    vartype : np.type that matches the length of the record length bytes
        default np.int32
    '''
    fi = open(fin , 'rb')

    # see if there are record length bytes in file fi
    skip_bytes = vartype(1).nbytes
    n1    = int.from_bytes(fi.read(skip_bytes), byteorder='little')
    fi.seek(n1, 1)
    n2    = int.from_bytes(fi.read(skip_bytes), byteorder='little')
    if n1 != n2:
        print("Ther are no reclen bytes in file {}".format(fin))
        return
    else:
        fi.seek(0, 2)
        nbytes = fi.tell()
        fi.seek(0, 0) # rewind

    # prepare output file name
    basename, ext = fin.split('.')
    fout = basename + '_out' + '.' + ext

    fo = open(fout,  'wb')

    passed = list(np.array(nbytes * np.arange(1, 21) / 20, dtype=int))
    perc   = list(np.array(100 * np.arange(1, 21) / 20, dtype=int))


    while fi.tell() < nbytes:
        n1    = int.from_bytes(fi.read(skip_bytes), byteorder='little')
        fo.write(fi.read(n1))
        n2    = int.from_bytes(fi.read(skip_bytes), byteorder='little')
        if n1 != n2:
            raise ValueError("Reclen bytes done match pos={}, n1={}, n2={}".format(fi.tell(), n1, n2))
        else:
            ipos = fi.tell()
            if ipos >= passed[0]:
                passed.pop(0)
                print(" {:3d}%".format(perc.pop(0)), end='')

    print("\nFile {} [size={}] --> {} [size={}]. Removed were {} record-length bytes."
             .format(fin, fi.tell(), fout, fo.tell(), fi.tell() - fo.tell()))

    fi.close()
    fo.close()
    return None