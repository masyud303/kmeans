import gdal
import numpy as np
import calckmeans as ck

inp = "F:\\kegiatan\\2019\\DATA\\input.tif"
out = "F:\\kegiatan\\2019\\DATA\\output.tif"

###############################################################################

def writearr(out, img, dtype, col, row, band, gt, proj, nodata):
    ## data type code : "uint8 and int8": 1, "uint16": 2, "int16": 3,
    ## "uint32": 4, "int32": 5, "float32": 6, "float64": 7
    if(dtype == 1):
        driver = gdal.GetDriverByName('GTIFF').Create(out, col, row, band, gdal.GDT_Byte)
    if(dtype == 2):
        driver = gdal.GetDriverByName('GTIFF').Create(out, col, row, band, gdal.GDT_UInt16)
    if(dtype == 4):
        driver = gdal.GetDriverByName('GTIFF').Create(out, col, row, band, gdal.GDT_UInt32)
    if(dtype == 6):
        driver = gdal.GetDriverByName('GTIFF').Create(out, col, row, band, gdal.GDT_Float32)
    if(dtype == 7):
        driver = gdal.GetDriverByName('GTIFF').Create(out, col, row, band, gdal.GDT_Float64)
    driver.SetGeoTransform(gt)
    driver.SetProjection(proj)
    for i in range(band):
        if(nodata != -1):
            driver.GetRasterBand(i+1).SetNoDataValue(nodata)
        if(band > 1):
            driver.GetRasterBand(i+1).WriteArray(img[i])
        if(band == 1):
            driver.GetRasterBand(i+1).WriteArray(img)

###############################################################################

igdal = gdal.Open(inp)
gt = igdal.GetGeoTransform()
proj = igdal.GetProjection()
row = igdal.RasterYSize
col = igdal.RasterXSize
iarr = igdal.ReadAsArray().astype(np.uint8)

k = 4
kmeans = np.array(ck.kmeans_calc(iarr, row, col, k), dtype=np.uint8)
writearr(out, kmeans, 1, col, row, 1, gt, proj, -1)
