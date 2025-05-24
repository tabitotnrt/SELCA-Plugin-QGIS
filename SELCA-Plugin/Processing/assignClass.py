from qgis.core import QgsRasterLayer
from qgis.PyQt.QtWidgets import QInputDialog
from osgeo import gdal
import numpy as np

def get_class_aliases(dialog):
    class_aliases = {}
    initialLC = dialog.initialLC.currentLayer()
    
    if isinstance(initialLC, QgsRasterLayer):
        unique_classes = get_unique_classes(initialLC.source())
        for cls in unique_classes:
            alias, ok = QInputDialog.getText(dialog, "Assign Class Alias", f"Enter alias for class {cls}:")
            if ok:
                class_aliases[cls] = alias
    return class_aliases

def get_unique_classes(raster_path):
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)
    
    # Dapatkan nilai NoData (jika ada)
    nodata_value = band.GetNoDataValue()
    
    # Dapatkan statistik dasar
    stats = band.GetStatistics(0, 1)
    if stats is None:
        band.ComputeStatistics(0)
        stats = band.GetStatistics(0, 1)
    min_val, max_val = stats[0], stats[1]
    
    # Jika range kecil, gunakan histogram
    if max_val - min_val < 10000:
        buckets = int(max_val - min_val + 1)
        histogram = band.GetHistogram(min_val, max_val+1, buckets, include_out_of_range=False, approx_ok=False)
        unique_classes = [int(min_val + idx) for idx, count in enumerate(histogram) if count > 0]
    else:
        # Untuk data besar, gunakan blok
        unique = set()
        xsize = band.XSize
        ysize = band.YSize
        x_block, y_block = band.GetBlockSize()
        
        for y in range(0, ysize, y_block):
            rows = min(y_block, ysize - y)
            for x in range(0, xsize, x_block):
                cols = min(x_block, xsize - x)
                array = band.ReadAsArray(x, y, cols, rows)
                unique.update(np.unique(array))
        
        unique_classes = sorted(unique)

    unique_classes = [cls for cls in unique_classes if cls != 0 and cls != nodata_value]

    return unique_classes
