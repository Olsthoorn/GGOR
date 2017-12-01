# -*- coding: utf-8 -*-
'''Read WGP (Water Gebieds Plan) data.
A WGP is an area to which the GGOR tool is to be applied.

The data are in a shape file. The attributes are in the dbf
file that belongs to the shapefile. The dbd file is read
using pandas to generate a DataFrame in pythong, which can then
be used to generate the GGOR MODFLOW model to simulate the
groundwater dynamices in all parces in the WGB in a single
run.
'''
import pandas as pd
import shapefile

dbfFile = 'AAN_GZK'

sf   = shapefile.Reader(dbfFile)
data = pd.DataFrame(sf.records())
hdr  = [f[0] for f in sf.fields[1:]]
data.columns = hdr
colDict = {h: h for h in hdr}

colDict = {'AANID': 'AANID',
 'Bodem': 'Bodem',
 'Bofek': 'Bofek',
 'FID1': 'FID1',
 'Gem_Cdek': 'Gem_Cdek',
 'Gem_Ddek': 'Gem_Ddek',
 'Gem_Kwel': 'Gem_Kwel',
 'Gem_Phi2': 'Gem_Phi2',
 'Gem_mAHN3': 'Gem_mAHN3',
 'Greppels': 'Greppels',
 'Grondsoort': 'Grondsoort',
 'LGN': 'LGN',
 'LGN_CODE': 'LGN_CODE',
 'Med_Cdek': 'Med_Cdek',
 'Med_Ddek': 'Med_Ddek',
 'Med_Kwel': 'Med_Kwel',
 'Med_Phi2': 'Med_Phi2',
 'Med_mAHN3': 'Med_mAHN3',
 'OBJECTID_1': 'OBJECTID_1',
 'Omtrek': 'Omtrek',
 'Oppervlak': 'Oppervlak',
 'Shape_Area': 'Shape_Area',
 'Shape_Leng': 'Shape_Leng',
 'Winterpeil': 'Winterpeil',
 'X_Midden': 'X_Midden',
 'Y_Midden': 'Y_Midden',
 'Zomerpeil': 'Zomerpeil'}
