# Derived from inspect.getsource(flopy.mf6.utils.postprocessing.get_structured_faceflows)

import numpy as np
import flopy.mf6.utils.binarygrid_util as fpb

def get_indices_to_pick_flf(grb_file=None):
    """Return the indices to pick the flow lower fact from the flowja array.
    
    This method only works for a structured MODFLOW 6 model.

    Parameters
    ----------
    flowja : ndarray
        flowja array for a structured MODFLOW 6 model
    grbfile : str
        MODFLOW 6 binary grid file path
    verbose: bool
        Write information to standard output

    Returns
    -------
    i_flf : ndarray (grb.nodes)
        node index
    j_flf : ndarray (number of conncections)
        index of corresponding flf in the flowja array
    """
    grb = fpb.MfGrdFile(grb_file)
    if grb.grid_type != "DIS":
        raise ValueError(
                "get_structured_faceflows method "
                "is only for structured DIS grids"
        )
    ia = grb.ia # CRS row pointers.
    ja = grb.ja # CRS column pointers.

    # get indices in flowja for flf
    j_flf = -np.ones(grb.nodes, dtype=int)
    for n in range(grb.nodes):
        i0, i1 = ia[n] + 1, ia[n + 1]        
        for j in range(i0, i1): # skips when i0 == i1 (no connected flows, inactive cell)
            jcol = ja[j]
            if jcol > n + 1 + grb.ncol:
                j_flf[n] = j
                break
                
    i_flf = np.arange(grb.nodes, dtype=int)
    
    return i_flf[j_flf > 0], j_flf[j_flf >= 0], grb


def get_flow_lower_face(flowja, grb_file=None, verbose=False):
    """Return flow_lower_face for all flowja (len nper).
    
    From the MODFLOW 6 flowja flows.
    This method only works for a structured MODFLOW 6 model.
    
    Parameters
    ----------
    flowja: list of nper np.arrays of connected cell flows
        from CBC.get_data(text='JA-FLOW-JA')
    grb: filename
        name of the binary flowpy grid file
        
    Returns
    -------
    flfs: np.array (grb.nodes, nper)
        The flow lower face 
    """
    nper = len(flowja)
    
    i_flf, j_flf, grb = get_indices_to_pick_flf(grb_file=grb_file)
    
    shape = (grb.nlay, grb.nrow, grb.ncol)
    flfs = dict()
    for iper, flwja in enumerate(flowja):
        flfs[iper] = np.zeros(grb.nodes)
        flfs[iper][i_flf] = -flwja.ravel()[j_flf]
        flfs[iper] = flfs[iper].reshape(shape)
    
    return flfs

