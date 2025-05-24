import os
import numpy as np
import pandas as pd
import rasterio
from urllib.parse import quote
from rasterstats import zonal_stats
from qgis.PyQt.QtCore import QAbstractTableModel, Qt, pyqtSignal, QPointF, QVariant
from qgis.PyQt.QtWidgets import QGraphicsScene, QComboBox, QGraphicsPixmapItem, QTableView, QWidget, QGraphicsView, QTextEdit,QFileDialog,QGraphicsTextItem,QGraphicsLineItem
from qgis.core import (QgsRasterLayer, QgsVectorLayer, QgsFeatureRequest, QgsTask, 
                      QgsApplication, QgsField, QgsFeature, QgsGeometry, QgsPointXY, QgsProject, QgsPalLayerSettings, QgsTextFormat, QgsVectorLayerSimpleLabeling,
                      QgsDistanceArea, QgsVectorFileWriter)
from qgis.gui import QgsFieldComboBox, QgsMapCanvas  # Add these imports
import matplotlib.pyplot as plt
from qgis.PyQt.QtGui import QPixmap  # Ensure QPixmap is imported
from rasterio.windows import from_bounds
from rasterio.mask import mask
from shapely.wkt import loads
from shapely.geometry import MultiPolygon
from PyQt5.QtGui import QPen, QColor
import time  # For yielding control
import datetime  # Added for timestamp
from PyQt5.QtCore import QModelIndex

class TableModel(QAbstractTableModel):
    def __init__(self, data, headers=None):
        super().__init__()
        self._data = data
        self._headers = headers or []

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return str(self._data[index.row()][index.column()])
        
    def rowCount(self, index=QModelIndex()):
        return len(self._data)

    def columnCount(self, index=QModelIndex()):
        return len(self._data[0]) if self._data else 0

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and self._headers:
            if orientation == Qt.Horizontal and section < len(self._headers):
                return self._headers[section]
            if orientation == Qt.Vertical:
                return str(section + 1)
        return None

# Background task for updating dependent tables
class UpdateDependentTablesTask(QgsTask):
    resultsComputed = pyqtSignal(pd.DataFrame, list)  # Include unique_values for region tables
    taskFailed = pyqtSignal(str)
    logMessage = pyqtSignal(str)

    def __init__(self, dialog, unique_values, dependent_layer_source, identifier_field):
        super().__init__('Update Dependent Tables', QgsTask.CanCancel)
        # Keep strong references
        self.dialog = dialog
        self.unique_values = unique_values
        self.dependent_layer_source = dependent_layer_source
        self.identifier_field = identifier_field
        self.master_data = None
        self.exception = None
        
        # Connect signals with direct connection
        self.resultsComputed.connect(lambda df, values: on_results_computed(self.dialog, df, values), 
                                   Qt.DirectConnection)
        self.taskFailed.connect(lambda msg: on_task_failed(self.dialog, msg),
                              Qt.DirectConnection)
        self.logMessage.connect(self.dialog.logWidget.append,
                              Qt.DirectConnection)

    def run(self):
        try:
            dataframes = []
            total_values = len(self.unique_values)
            for idx, value in enumerate(self.unique_values):
                if self.isCanceled():
                    return False
                df = self.update_dependent_table(value)
                if df is None:
                    self.logMessage.emit(f"Skipping update for mask value {value} due to error.")
                    continue
                dataframes.append(df)
                if idx % 10 == 0 or idx == total_values - 1:
                    self.setProgress((idx + 1) / total_values * 50)  # Update progress for table updates
            self.master_data = pd.concat(dataframes, ignore_index=True)
            return True
        except Exception as e:
            self.exception = str(e)
            return False

    def update_dependent_table(self, mask_value):
        initialLC = self.dialog.initialLC.currentLayer().source()
        finalLC = self.dialog.finalLC.currentLayer().source()
        additionalLCs = [combo.currentLayer().source() for combo in self.dialog.landCoverComboBoxes]
        mask_layer_source = self.dependent_layer_source
        identifier_field = self.identifier_field

        initial_stats = self.compute_area_statistics_by_mask(initialLC, mask_layer_source, mask_value, identifier_field)
        if not isinstance(initial_stats, dict):
            self.logMessage.emit(f"Error: Invalid initial_stats for mask value {mask_value}")
            return None
        final_stats = self.compute_area_statistics_by_mask(finalLC, mask_layer_source, mask_value, identifier_field)
        additional_stats = [
            self.compute_area_statistics_by_mask(lc, mask_layer_source, mask_value, identifier_field)
            for lc in additionalLCs
        ]

        headers = ["Region", "Time Stamp"] + [self.dialog.classAliases.get(cls, str(cls)) for cls in sorted(initial_stats.keys())]
        data = []

        initial_row = [mask_value, self.dialog.initialAlias.text() or "Initial"]
        for cls in sorted(initial_stats.keys()):
            initial_row.append(f"{initial_stats.get(cls, 0):,.2f}")
        data.append(initial_row)

        for i, stats in enumerate(additional_stats):
            row = [mask_value, self.dialog.aliasLineEdits[i].text() or f"LC_{i + 2}"]
            for cls in sorted(initial_stats.keys()):
                row.append(f"{stats.get(cls, 0):,.2f}")
            data.append(row)

        final_row = [mask_value, self.dialog.finalAlias.text() or "Final"]
        for cls in sorted(initial_stats.keys()):
            final_row.append(f"{final_stats.get(cls, 0):,.2f}")
        data.append(final_row)

        headers.append(self.dialog.aliasDependent.text() or "Dependent")
        dependent_values = self.get_dependent_values(mask_value)
        if len(dependent_values) != len(data):
            self.logMessage.emit(f"Error: Mismatched dependent values for mask value {mask_value}")
            return None
        for i, row in enumerate(data):
            row.append(dependent_values[i])

        return pd.DataFrame(data, columns=headers)

    def compute_area_statistics_by_mask(self, raster_path, mask_layer_source, mask_value, identifier_field):
        try:
            mask_layer = QgsVectorLayer(mask_layer_source, "temp_mask", "ogr")
            if not mask_layer.isValid():
                raise ValueError("Invalid mask layer")

            with rasterio.open(raster_path) as src:
                pixel_area = abs(src.transform[0] * src.transform[5]) / 1e6
                mask_request = QgsFeatureRequest().setFilterExpression(f'"{identifier_field}" = \'{mask_value}\'')
                mask_geometries = [f.geometry().asWkt() for f in mask_layer.getFeatures(mask_request)]

                if not mask_geometries:
                    return {}

                stats = zonal_stats(
                    mask_geometries, raster_path, stats="count", categorical=True,
                    nodata=src.nodata, all_touched=True
                )
                area_stats = {}
                for stat in stats:
                    for class_value, count in stat.items():
                        if class_value == 'count':
                            continue
                        area_stats[class_value] = area_stats.get(class_value, 0) + count * pixel_area/1e6
                return area_stats
        except Exception as e:
            self.logMessage.emit(f"Error in compute_area_statistics_by_mask: {str(e)}")
            return {}

    def compute_city_center_shift_by_mask(self, built_up_value):

        try:
            # Get raster paths from dialog in chronological order: oldest to newest
            initialLC = self.dialog.initialLC.currentLayer().source()
            finalLC = self.dialog.finalLC.currentLayer().source()
            additionalLCs = [combo.currentLayer().source() for combo in self.dialog.landCoverComboBoxes]
            raster_paths = [initialLC] + additionalLCs + [finalLC]
            self.logMessage.emit(f"Processing {len(raster_paths)} rasters")

            # Get mask layer source and identifier field
            mask_layer_source = self.dependent_layer_source
            identifier_field = self.identifier_field

            # Load mask layer
            mask_layer = QgsVectorLayer(mask_layer_source, "temp_mask", "ogr")
            if not mask_layer.isValid():
                raise ValueError("Invalid mask layer")

            # Store results for all cities
            city_centroids = {}

            # Get all features and determine batch size
            features = list(mask_layer.getFeatures())
            total_features = len(features)
            if total_features == 0:
                self.logMessage.emit("No features found in mask layer")
                return {}

            # Set a small batch size for responsiveness with small datasets
            batch_size = min(2, total_features)  # Process 1-2 cities at a time
            total_batches = (total_features + batch_size - 1) // batch_size
            self.logMessage.emit(f"Processing {total_features} features in {total_batches} batches of {batch_size}")

            # Process features in small batches
            for batch_num in range(total_batches):
                if self.isCanceled():
                    self.logMessage.emit("Computation canceled by user")
                    return {}
                
                start = batch_num * batch_size
                end = min(start + batch_size, total_features)
                batch = features[start:end]

                for feature in batch:
                    mask_value = feature[identifier_field]

                    # Preprocess geometry once per city
                    t_start = time.time()
                    mask_request = QgsFeatureRequest().setFilterExpression(f'"{identifier_field}" = \'{mask_value}\'')
                    mask_geometries = [loads(f.geometry().asWkt()) for f in mask_layer.getFeatures(mask_request)]
                    if not mask_geometries:
                        self.logMessage.emit(f"No geometry found for city '{mask_value}' in field '{identifier_field}'")
                        city_centroids[mask_value] = []
                        continue

                    # Compute bounding box once
                    if len(mask_geometries) > 1:
                        minx, miny, maxx, maxy = MultiPolygon(mask_geometries).bounds
                    else:
                        minx, miny, maxx, maxy = mask_geometries[0].bounds
                    self.logMessage.emit(f"Geometry prep for {mask_value} took {time.time() - t_start:.2f}s")

                    # Store centroids for this city
                    points = []

                    # Process each raster
                    for i, raster_path in enumerate(raster_paths):
                        layer_name = f"Layer_{i+1}"
                        t_raster_start = time.time()

                        with rasterio.open(raster_path) as src:
                            # Define window
                            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

                            # Apply exact geometry mask
                            masked_data, masked_transform = mask(
                                src, shapes=mask_geometries, crop=True, nodata=src.nodata, all_touched=True
                            )
                            masked_data = masked_data[0]

                            # Find built-up pixels
                            rows, cols = np.where(masked_data == built_up_value)
                            if len(rows) == 0:
                                self.logMessage.emit(f"No built-up areas (value {built_up_value}) found in {raster_path} for city '{mask_value}'")
                                continue

                            # Calculate centroid
                            mean_row, mean_col = np.mean(rows), np.mean(cols)
                            centroid_x, centroid_y = masked_transform * (mean_col, mean_row)

                            # Store point
                            points.append({
                                "layer_name": layer_name,
                                "x": centroid_x,
                                "y": centroid_y
                            })
                            self.logMessage.emit(f"Computed centroid for {mask_value} on {layer_name}: ({centroid_x}, {centroid_y}) in {time.time() - t_raster_start:.2f}s")

                    city_centroids[mask_value] = points

                # Update progress and yield control
                progress = 50 + ((batch_num + 1) / total_batches) * 50  # Update progress for centroid computation
                self.setProgress(progress)
                self.logMessage.emit(f"Completed batch {batch_num + 1}/{total_batches}")
                time.sleep(0.01)  # Brief pause to yield control to QGIS

            self.logMessage.emit(f"City centroids computed: {city_centroids}")
            return city_centroids

        except Exception as e:
            self.logMessage.emit(f"Error in compute_city_center_shift_by_mask: {str(e)}")
            return {}
        
    def compute_city_center_shift_full_raster(self, built_up_value):
        """
        Compute the centroid of built-up areas across full raster extents over time.
        
        Args:
            built_up_value: The raster value representing built-up areas.
        
        Returns:
            dict: A dictionary with a single key 'global' mapping to a list of centroid coordinates per raster.
        """
        try:
            # Get raster paths from dialog in chronological order: oldest to newest
            initialLC = self.dialog.initialLC.currentLayer().source()
            finalLC = self.dialog.finalLC.currentLayer().source()
            additionalLCs = [combo.currentLayer().source() for combo in self.dialog.landCoverComboBoxes]
            raster_paths = [initialLC] + additionalLCs + [finalLC]
            self.logMessage.emit(f"Processing {len(raster_paths)} rasters")

            # Store results as a single global set of centroids
            city_centroids = {"global": []}
            total_rasters = len(raster_paths)

            # Process each raster
            for i, raster_path in enumerate(raster_paths):
                if self.isCanceled():
                    self.logMessage.emit("Computation canceled by user")
                    return {}

                layer_name = f"Layer_{i+1}"
                t_raster_start = time.time()

                with rasterio.open(raster_path) as src:
                    # Read the entire raster (no window or mask)
                    raster_data = src.read(1)  # First band
                    transform = src.transform

                    # Find built-up pixels across the entire raster
                    rows, cols = np.where(raster_data == built_up_value)
                    if len(rows) == 0:
                        self.logMessage.emit(f"No built-up areas (value {built_up_value}) found in {raster_path}")
                        continue

                    # Calculate centroid in pixel coordinates
                    mean_row, mean_col = np.mean(rows), np.mean(cols)

                    # Convert to geographic coordinates
                    centroid_x, centroid_y = transform * (mean_col, mean_row)

                    # Store point
                    city_centroids["global"].append({
                        "layer_name": layer_name,
                        "x": centroid_x,
                        "y": centroid_y
                    })
                    self.logMessage.emit(f"Computed centroid for {layer_name}: ({centroid_x}, {centroid_y}) in {time.time() - t_raster_start:.2f}s")

                # Update progress
                progress = ((i + 1) / total_rasters) * 100
                self.setProgress(progress)
                self.logMessage.emit(f"Completed raster {i + 1}/{total_rasters}")
                time.sleep(0.01)  # Brief pause to yield control to QGIS

            self.logMessage.emit(f"Global centroids computed: {city_centroids}")
            return city_centroids

        except Exception as e:
            self.logMessage.emit(f"Error in compute_city_center_shift_full_raster: {str(e)}")
            return {}

    def get_dependent_values(self, mask_value):
        mask_layer = QgsVectorLayer(self.dependent_layer_source, "temp_mask", "ogr")
        dependent_values = []
        combos = [self.dialog.initialDependent] + self.dialog.dependentVariableComboBoxes + [self.dialog.finalDependent]
        for combo in combos:
            field_name = combo.currentText()
            request = QgsFeatureRequest().setFilterExpression(f'"{self.identifier_field}" = \'{mask_value}\'')
            for feature in mask_layer.getFeatures(request):
                dependent_values.append(feature[field_name] if field_name in feature.fields().names() else 0)
                break
            else:
                dependent_values.append(0)
        return dependent_values

    def finished(self, result):
        if result:
            self.resultsComputed.emit(self.master_data, self.unique_values)

        else:
            self.taskFailed.emit(self.exception or "Task was canceled.")

# Main function to start the task
def update_graphics_perkota(dialog, df, tab):
    if df.empty:
        dialog.logWidget.append("Error: DataFrame is empty. Skipping plot update.")
        return

    classes = df.columns[2:-1]  # Exclude 'Region', 'Time Stamp', and the dependent variable
    timestamps = df['Time Stamp']
    dependent_field = dialog.aliasDependent.text() if dialog.aliasDependent else "Dependent"
    dependent_values = df[dependent_field]
    
    # Debugging logs
    dialog.logWidget.append(f"Available columns: {df.columns.tolist()}")
    dialog.logWidget.append(f"Attempting to use columns: {classes.tolist()}")
    
    try:
        data = df[classes].replace({',': ''}, regex=True).apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if data.empty or data.isnull().values.any():
            dialog.logWidget.append("Error: No valid numeric data available for plotting.")
            return
        
        # Create figure with exact 2:1 aspect ratio
        fig = plt.figure(figsize=(12, 6))  # 2:1 aspect ratio
        fig.set_size_inches(12, 6, forward=True)
        ax1 = fig.add_subplot(111)
        
        x = np.arange(len(timestamps))  # Use the index for the x-axis
        bottom = np.zeros(len(timestamps))

        for cls in classes:
            ax1.bar(x, data[cls], bottom=bottom, label=cls)
            bottom += data[cls]

        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Total Land Area (km²)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(timestamps, rotation=45, ha='right')
        ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))

        ax2 = ax1.twinx()
        ax2.plot(x, dependent_values, color='red', marker='o', linestyle='-', label=dependent_field)
        ax2.set_ylabel(dependent_field)
        ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))

        graphics_perkota = tab.findChild(QGraphicsView, f"graphics_{tab.objectName().split('_')[-1]}")
    
        if graphics_perkota is None:
            dialog.logWidget.append(f"Error: No QGraphicsView found for {tab.objectName()}")
            return

        # Ensure the QGraphicsView has a scene
        if graphics_perkota.scene() is None:
            graphics_perkota.setScene(QGraphicsScene())

        graphics_perkota.scene().clear()

        # Ensure the /tmp directory exists
        tmp_dir = "/tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        # Save the figure
        fig.savefig(os.path.join(tmp_dir, "temp_perkota_chart.png"), bbox_inches='tight')

        # Load the saved image and add to the scene
        image_path = os.path.join(tmp_dir, "temp_perkota_chart.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            dialog.logWidget.append(f"Error: Failed to load image {image_path}")
            return

        item = QGraphicsPixmapItem(pixmap)
        item.setTransformationMode(Qt.SmoothTransformation)
        graphics_perkota.scene().addItem(item)

        # Ensure items exist before calling itemsBoundingRect()
        if not graphics_perkota.scene().items():
            dialog.logWidget.append("Warning: No items in scene, skipping fitInView.")
            return

        # Force 2:1 aspect ratio
        graphics_perkota.setMinimumSize(600, 300)  # Base size with 2:1 ratio
        graphics_perkota.setFixedHeight(graphics_perkota.width() // 2)  # Force height to be half the width

        graphics_perkota.fitInView(graphics_perkota.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

        # Turn off scroll bars
        graphics_perkota.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        graphics_perkota.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Adjust the scene rectangle to fit the item
        graphics_perkota.setSceneRect(graphics_perkota.scene().itemsBoundingRect())

        # Connect resize event to maintain ratio
        graphics_perkota.resizeEvent = lambda event: graphics_perkota.fitInView(
            graphics_perkota.scene().itemsBoundingRect(), Qt.KeepAspectRatio
        )

        # Write important statistics to textPerkota
        text_perkota = tab.findChild(QTextEdit, f"text_{tab.objectName().split('_')[-1]}")
        if text_perkota:
            stats_text = f"Statistics for {tab.objectName().split('_')[-1]}:\n"
            stats_text += f"Total Land Area (Initial): {data.iloc[0, 2:].sum():,.2f} km²\n"
            stats_text += f"Total Land Area (Final): {data.iloc[-1, 2:].sum():,.2f} km²\n"
            stats_text += f"Dependent Variable (Initial): {dependent_values.iloc[0]:,.2f}\n"
            stats_text += f"Dependent Variable (Final): {dependent_values.iloc[-1]:,.2f}\n"
            text_perkota.setText(stats_text)

    except Exception as e:
        dialog.logWidget.append(f"Error while plotting: {str(e)}")

def update_graphics_sum(dialog, master_data):
    if master_data.empty:
        dialog.logWidget.append("Error: Master DataFrame is empty. Skipping plot update.")
        return

    dialog.logWidget.append(f"Master data for graphicsSum: {master_data.head().to_string()}")
    classes = master_data.columns[2:-1]
    timestamps = sorted(master_data['Time Stamp'].unique(), 
                        key=lambda x: (x != "initialAlias", x == "finalAlias", x))
    dependent_field = dialog.aliasDependent.text() if dialog.aliasDependent else "Dependent"
    dependent_values = master_data.groupby('Time Stamp')[dependent_field].sum().reindex(timestamps)

    try:
        data = master_data.groupby(['Time Stamp', 'Region'])[classes].sum().replace({',': ''}, regex=True).apply(pd.to_numeric, errors='coerce').fillna(0).reindex(timestamps, level=0)
        if data.empty or data.isnull().values.any():
            dialog.logWidget.append("Error: No valid numeric data available for plotting.")
            return
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        x = np.arange(len(timestamps))
        bottom = np.zeros(len(timestamps))

        for cls in classes:
            ax1.bar(x, data[cls].groupby('Time Stamp').sum(), bottom=bottom, label=cls)
            bottom += data[cls].groupby('Time Stamp').sum()

        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Total Land Area (km²)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(timestamps, rotation=45, ha='right')
        ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))

        ax2 = ax1.twinx()
        ax2.plot(x, dependent_values, color='red', marker='o', linestyle='-', label=dependent_field)
        ax2.set_ylabel(dependent_field)
        ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))

        if dialog.graphicsSum is None:
            dialog.logWidget.append("Error: dialog.graphicsSum is None")
            return

        if dialog.graphicsSum.scene() is None:
            dialog.graphicsSum.setScene(QGraphicsScene())
        dialog.graphicsSum.scene().clear()

        tmp_dir = "/tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        image_path = os.path.join(tmp_dir, "temp_sum_chart.png")
        fig.savefig(image_path, bbox_inches='tight')
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            dialog.logWidget.append(f"Error: Failed to load image {image_path}")
            return

        item = QGraphicsPixmapItem(pixmap)
        item.setTransformationMode(Qt.SmoothTransformation)
        dialog.graphicsSum.scene().addItem(item)

        if not dialog.graphicsSum.scene().items():
            dialog.logWidget.append("Warning: No items in scene, skipping fitInView.")
            return

        dialog.graphicsSum.fitInView(dialog.graphicsSum.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        dialog.graphicsSum.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        dialog.graphicsSum.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        dialog.graphicsSum.setSceneRect(dialog.graphicsSum.scene().itemsBoundingRect())

    except Exception as e:
        dialog.logWidget.append(f"Error while plotting graphicsSum: {str(e)}")

    dialog.logWidget.append("graphicsSum update completed.")

def update_dependent_tables(dialog):
    dialog.progressBar.setValue(0)
    dialog.logWidget.clear()
    dialog.logWidget.append("Starting update of dependent tables...")

    dependent_layer = dialog.layerDependent.currentLayer()
    if not isinstance(dependent_layer, QgsVectorLayer):
        dialog.logWidget.append("Error: Please select a valid dependent vector layer.")
        return

    identifier_field = dialog.identifierField.currentText()
    unique_values = list(dependent_layer.uniqueValues(dependent_layer.fields().indexFromName(identifier_field)))
    dependent_layer_source = dependent_layer.source()

    # Create and store task
    task = UpdateDependentTablesTask(dialog, unique_values, dependent_layer_source, identifier_field)
    
    # Store strong reference to prevent garbage collection
    dialog.current_task = task
    
    # Connect progress with direct connection
    task.progressChanged.connect(lambda: dialog.progressBar.setValue(int(task.progress())),
                               Qt.DirectConnection)
    
    # Add task to manager
    QgsApplication.taskManager().addTask(task)
    dialog.logWidget.append("Task started.")

def update_graphics_center(dialog, city_centroids):
    """
    Update graphicsCenter for each city tab with centroid data, including arrows between consecutive points.
    Add Google Hybrid imagery as a basemap.
    """
    # Extract CRS from initial land cover raster
    initialLC = dialog.initialLC.currentLayer().source()
    raster_layer = QgsRasterLayer(initialLC, "Initial Land Cover")

    if not raster_layer.isValid():
        dialog.logWidget.append("Error: Unable to extract EPSG from Initial Land Cover raster!")
        return

    landcover_crs = raster_layer.crs()
    landcover_epsg = landcover_crs.authid()

    if len(city_centroids) == 0:
        dialog.logWidget.append("No city centroid data to visualize.")
        return

    # Generate a timestamp for unique layer naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, (city, points) in enumerate(city_centroids.items()):
        tab = dialog.tabWidgetAbangku.findChild(QWidget, f"tab_{city}")
        if not tab:
            dialog.logWidget.append(f"Tab not found for city {city} - skipping graphicsCenter update.")
            continue

        graphics_center = tab.findChild(QgsMapCanvas, f"graphicsCenter_{city}")
        if not graphics_center:
            dialog.logWidget.append(f"graphicsCenter not found for city {city} - skipping update.")
            continue
        
        # Add Google Hybrid imagery as a basemap
        google_hybrid_url = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
        zmin=0
        zmax=20
        crs='EPSG:4326'
        uri = f'type=xyz&url={google_hybrid_url}&zmax={zmax}&zmin={zmin}$crs={crs}'
        google_hybrid_layer = QgsRasterLayer(uri, "OSM Basemap", "wms")
        if not google_hybrid_layer.isValid():
            dialog.logWidget.append("Error: Unable to load Google Hybrid imagery.")
        else:
            QgsProject.instance().addMapLayer(google_hybrid_layer)

        # Create a vector layer for points with a descriptive name
        point_layer_name = f"{city}_Centroids_{timestamp}"
        point_layer = QgsVectorLayer(f"Point?crs={landcover_epsg}", point_layer_name, "memory")
        pr = point_layer.dataProvider()
        pr.addAttributes([QgsField("layer_name", QVariant.String)])
        point_layer.updateFields()

        features = []
        num_points = len(points)

        for i, point in enumerate(points):
            alias = (
                dialog.initialAlias.text() if i == 0 and dialog.initialAlias else
                dialog.finalAlias.text() if i == num_points - 1 and dialog.finalAlias else
                dialog.aliasLineEdits[i - 1].text() if i - 1 < len(dialog.aliasLineEdits) else f"LC_{i + 1}"
            )

            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point["x"], point["y"])))
            feature.setAttributes([alias])
            features.append(feature)

            point["layer_name"] = alias

        pr.addFeatures(features)
        point_layer.updateExtents()

        # Enable Labeling for 'layer_name' column (Points)
        point_label_settings = QgsPalLayerSettings()
        point_label_settings.fieldName = "layer_name"
        point_label_settings.enabled = True

        point_text_format = QgsTextFormat()
        point_text_format.setSize(10)
        point_label_settings.setFormat(point_text_format)

        point_layer.setLabelsEnabled(True)
        point_layer.setLabeling(QgsVectorLayerSimpleLabeling(point_label_settings))
        point_layer.triggerRepaint()

        QgsProject.instance().addMapLayer(point_layer)

        # Create a vector layer for lines with a descriptive name
        line_layer_name = f"{city}_Centroid_Shifts_{timestamp}"
        line_layer = QgsVectorLayer(f"LineString?crs={landcover_epsg}", line_layer_name, "memory")
        line_pr = line_layer.dataProvider()
        line_pr.addAttributes([
            QgsField("layer_name", QVariant.String),
            QgsField("length_km", QVariant.Double)
        ])
        line_layer.updateFields()

        dist_calc = QgsDistanceArea()
        dist_calc.setSourceCrs(landcover_crs, QgsProject.instance().transformContext())
        dist_calc.setEllipsoid(landcover_crs.ellipsoidAcronym())

        prev_point = None
        for point in points:
            if prev_point:
                line = QgsFeature()
                line_geom = QgsGeometry.fromPolylineXY([
                    QgsPointXY(prev_point["x"], prev_point["y"]),
                    QgsPointXY(point["x"], point["y"])
                ])
                length = dist_calc.measureLength(line_geom) / 1000

                line.setGeometry(line_geom)
                line.setAttributes([point["layer_name"], round(length, 3)])
                line_pr.addFeature(line)

            prev_point = point

        line_layer.updateExtents()

        # Enable Labeling for 'length_km' column (Lines)
        line_label_settings = QgsPalLayerSettings()
        line_label_settings.fieldName = "length_km"
        line_label_settings.enabled = True

        line_text_format = QgsTextFormat()
        line_text_format.setSize(10)
        line_label_settings.setFormat(line_text_format)
        line_label_settings.placement = QgsPalLayerSettings.Line
        line_layer.setLabelsEnabled(True)
        line_layer.setLabeling(QgsVectorLayerSimpleLabeling(line_label_settings))
        line_layer.triggerRepaint()

        QgsProject.instance().addMapLayer(line_layer)
        graphics_center.setDestinationCrs(landcover_crs)
        graphics_center.setLayers([point_layer, line_layer, google_hybrid_layer])
        graphics_center.setExtent(point_layer.extent())
        graphics_center.refresh()

        dialog.progressBar.setValue(int(50 + ((idx + 1) / len(city_centroids)) * 50))

def on_results_computed(dialog, master_data, unique_values):
    """Update UI with results for each region and the summary..."""
    dialog.master_data = master_data  # Ensure master_data is populated
    dialog.logWidget.append("Updating tables for regions and summary...")
    
    # Convert all NaN values to 0
    master_data = master_data.fillna(0)

    # Ensure dependent variable field is at the end
    dependent_field = dialog.aliasDependent.text() if dialog.aliasDependent else "Dependent"
    columns = [col for col in master_data.columns if col != dependent_field] + [dependent_field]
    master_data = master_data[columns]

    # Single attempt to populate the observedField combo box
    try:
        observer_field = dialog.findChild(QComboBox, "observedField")
        dialog.logWidget.append(f"Looking for observedField: {observer_field is not None}")
        
        if observer_field is not None:
            observer_field.clear()
            numeric_columns = [col for col in master_data.columns 
                             if col not in ['Region', 'Time Stamp', dependent_field]]
            observer_field.addItems(numeric_columns)
            dialog.logWidget.append(f"Added columns to observedField: {numeric_columns}")
        else:
            dialog.logWidget.append("Warning: observedField combo box not found in UI")
            dialog.logWidget.append(f"Available widgets: {[w.objectName() for w in dialog.findChildren(QComboBox)]}")
    except Exception as e:
        dialog.logWidget.append(f"Error updating observedField: {str(e)}")

    # Update individual region tables
    for value in unique_values:
        # Filter data for this region
        region_data = master_data[master_data["Region"] == value]
        tab = dialog.tabWidgetAbangku.findChild(QWidget, f"tab_{value}")
        
        if tab:
            table = tab.findChild(QTableView, f"table_{value}")
            if table:
                # Update existing table
                model = TableModel(region_data.values.tolist(), region_data.columns.tolist())
                table.setModel(model)
                table.resizeColumnsToContents()
                table.resizeRowsToContents()
                dialog.logWidget.append(f"Updated table for region {value}")
            else:
                # Create new table if it doesn’t exist
                table = QTableView(tab)
                table.setObjectName(f"table_{value}")
                model = TableModel(region_data.values.tolist(), region_data.columns.tolist())
                table.setModel(model)
                tab.layout().addWidget(table)  # Assumes tab has a layout
                table.resizeColumnsToContents()
                table.resizeRowsToContents()
                dialog.logWidget.append(f"Created and updated table for region {value}")
            update_graphics_perkota(dialog, region_data, tab)  # Update graphicsPerkota
        else:
            dialog.logWidget.append(f"Tab not found for region {value} - skipping table update")

    # Update summary tab using stored references
    tab_sum = dialog.tabWidgetAbangku.findChild(QWidget, "tabSum")
    if tab_sum:
        if dialog.tableSum:
            model = TableModel(master_data.values.tolist(), master_data.columns.tolist())
            dialog.tableSum.setModel(model)
            dialog.tableSum.resizeColumnsToContents()
            dialog.tableSum.resizeRowsToContents()
            dialog.logWidget.append("Master data visualized on tableSum.")
        if dialog.graphicsSum:
            dialog.logWidget.append("Attempting to update graphicsSum...")
            update_graphics_sum(dialog, master_data)
            dialog.logWidget.append("GraphicsSum update attempted.")
        else:
            dialog.logWidget.append("Error: graphicsSum reference not set.")
    else:
        dialog.logWidget.append("Error: tabSum not found.")
        
    dialog.progressBar.setValue(100)
    dialog.logWidget.append("Dependent tables updated.")

    built_up_value = int(dialog.findChild(QComboBox, "observedClass").currentText())
    global_centroids = dialog.current_task.compute_city_center_shift_full_raster(built_up_value) if dialog.current_task else {}
    dialog.logWidget.append(f"Global centroids computed: {global_centroids}")
    if dialog.graphicsCenterSum:
        dialog.logWidget.append("Attempting to update graphicsCenterSum...")
        update_graphics_center_global(dialog, global_centroids)
        dialog.logWidget.append("GraphicsCenterSum update attempted.")
    else:
        dialog.logWidget.append("Error: graphicsCenterSum reference not set.")
    city_centroids = dialog.current_task.compute_city_center_shift_by_mask(built_up_value) if dialog.current_task else {}
    update_graphics_center(dialog, city_centroids)

    # Ensure all tabs are updated
    for i in range(dialog.tabWidgetAbangku.count()):
        dialog.tabWidgetAbangku.setCurrentIndex(i)
        tab = dialog.tabWidgetAbangku.widget(i)
        if tab:
            graphics_perkota = tab.findChild(QGraphicsView, f"graphics_{tab.objectName().split('_')[-1]}")
            if graphics_perkota and graphics_perkota.scene():
                graphics_perkota.fitInView(graphics_perkota.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

def on_task_failed(dialog, message):
    dialog.logWidget.append(f"Task failed: {message}")
    dialog.progressBar.setValue(0)

def update_graphics_center_global(dialog, city_centroids):
    """
    Update graphicsCenterSum with global centroid data, including arrows between consecutive points.
    Add Google Hybrid imagery as a basemap.
    """
    if len(city_centroids) == 0 or "global" not in city_centroids or len(city_centroids["global"]) == 0:
        dialog.logWidget.append("No global centroid data to visualize.")
        return

    initialLC = dialog.initialLC.currentLayer().source()
    raster_layer = QgsRasterLayer(initialLC, "Initial Land Cover")
    if not raster_layer.isValid():
        dialog.logWidget.append("Error: Unable to extract EPSG from Initial Land Cover raster!")
        return

    landcover_crs = raster_layer.crs()
    landcover_epsg = landcover_crs.authid()

    if dialog.graphicsCenterSum is None:
        dialog.logWidget.append("Error: graphicsCenterSum reference not set.")
        return

    # Add Google Hybrid imagery as a basemap
    google_hybrid_url = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
    zmin=0
    zmax=20
    crs='EPSG:4326'
    uri = f'type=xyz&url={google_hybrid_url}&zmax={zmax}&zmin={zmin}$crs={crs}'
    google_hybrid_layer = QgsRasterLayer(uri, "OSM Basemap", "wms")
    if not google_hybrid_layer.isValid():
        dialog.logWidget.append("Error: Unable to load Google Hybrid imagery.")
    else:
        QgsProject.instance().addMapLayer(google_hybrid_layer)

    # Generate a timestamp for unique layer naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a vector layer for points with a descriptive name
    point_layer_name = f"Global_Centroids_{timestamp}"
    point_layer = QgsVectorLayer(f"Point?crs={landcover_epsg}", point_layer_name, "memory")
    pr = point_layer.dataProvider()
    pr.addAttributes([QgsField("layer_name", QVariant.String)])
    point_layer.updateFields()

    points = city_centroids["global"]
    features = []
    num_points = len(points)

    for i, point in enumerate(points):
        alias = (
            dialog.initialAlias.text() if i == 0 and dialog.initialAlias else
            dialog.finalAlias.text() if i == num_points - 1 and dialog.finalAlias else
            dialog.aliasLineEdits[i - 1].text() if i - 1 < len(dialog.aliasLineEdits) else f"LC_{i + 1}"
        )
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point["x"], point["y"])))
        feature.setAttributes([alias])
        features.append(feature)
        point["layer_name"] = alias

    pr.addFeatures(features)
    point_layer.updateExtents()

    point_label_settings = QgsPalLayerSettings()
    point_label_settings.fieldName = "layer_name"
    point_label_settings.enabled = True
    point_text_format = QgsTextFormat()
    point_text_format.setSize(10)
    point_label_settings.setFormat(point_text_format)
    point_layer.setLabelsEnabled(True)
    point_layer.setLabeling(QgsVectorLayerSimpleLabeling(point_label_settings))
    point_layer.triggerRepaint()

    QgsProject.instance().addMapLayer(point_layer)

    # Create a vector layer for lines with a descriptive name
    line_layer_name = f"Global_Centroid_Shifts_{timestamp}"
    line_layer = QgsVectorLayer(f"LineString?crs={landcover_epsg}", line_layer_name, "memory")
    line_pr = line_layer.dataProvider()
    line_pr.addAttributes([
        QgsField("layer_name", QVariant.String),
        QgsField("length_km", QVariant.Double)
    ])
    line_layer.updateFields()

    dist_calc = QgsDistanceArea()
    dist_calc.setSourceCrs(landcover_crs, QgsProject.instance().transformContext())
    dist_calc.setEllipsoid(landcover_crs.ellipsoidAcronym())

    prev_point = None
    for point in points:
        if prev_point:
            line = QgsFeature()
            line_geom = QgsGeometry.fromPolylineXY([
                QgsPointXY(prev_point["x"], prev_point["y"]),
                QgsPointXY(point["x"], point["y"])
            ])
            length = dist_calc.measureLength(line_geom) / 1000
            line.setGeometry(line_geom)
            line.setAttributes([point["layer_name"], round(length, 3)])
            line_pr.addFeature(line)
        prev_point = point

    line_layer.updateExtents()

    line_label_settings = QgsPalLayerSettings()
    line_label_settings.fieldName = "length_km"
    line_label_settings.enabled = True
    line_text_format = QgsTextFormat()
    line_text_format.setSize(10)
    line_label_settings.setFormat(line_text_format)
    line_label_settings.placement = QgsPalLayerSettings.Line
    line_layer.setLabelsEnabled(True)
    line_layer.setLabeling(QgsVectorLayerSimpleLabeling(line_label_settings))
    line_layer.triggerRepaint()

    QgsProject.instance().addMapLayer(line_layer)

    dialog.graphicsCenterSum.setLayers([google_hybrid_layer, point_layer, line_layer])
    dialog.graphicsCenterSum.setExtent(point_layer.extent())
    dialog.graphicsCenterSum.refresh()

    dialog.logWidget.append("graphicsCenterSum rendering completed.")

def export_dependent_data(dialog):
    """Export PNGs, CSVs, and shapefiles for dependent data into subfolders with descriptive names."""
    if not hasattr(dialog, 'master_data') or dialog.master_data.empty:
        dialog.logWidget.append("Error: Run dependent analysis first to populate data.")
        dialog.progressBar.setValue(0)
        return

    dialog.logWidget.append("Exporting dependent data...")
    dialog.progressBar.setValue(0)

    base_folder = QFileDialog.getExistingDirectory(dialog, "Select Base Folder to Save Dependent Data")
    if not base_folder:
        dialog.logWidget.append("Export canceled: No folder selected.")
        dialog.progressBar.setValue(0)
        return

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        all_tabs = dialog.tabWidgetAbangku.findChildren(QWidget)
        dialog.logWidget.append(f"All widgets in tabWidgetAbangku: {[w.objectName() for w in all_tabs]}")
        unique_values = [w.objectName().split('_')[-1] for w in all_tabs 
                         if w.objectName().startswith("tab_") and w.objectName() != "tabSum"]
        dialog.logWidget.append(f"Found {len(unique_values)} city tabs: {unique_values}")
        total_steps = len(unique_values) * 3 + 3  # PNG + CSV + shapefiles per city, plus global PNG + CSV + shapefiles
        step = 0

        # Step 1: Export for individual cities
        for value in unique_values:
            tab = dialog.tabWidgetAbangku.findChild(QWidget, f"tab_{value}")
            if not tab:
                dialog.logWidget.append(f"Tab not found for region {value} - skipping export.")
                continue
            dialog.logWidget.append(f"Processing tab_{value}")

            # Create subfolder for this city
            city_folder = os.path.join(base_folder, str(value))
            os.makedirs(city_folder, exist_ok=True)
            dialog.logWidget.append(f"Created subfolder: {city_folder}")

            # Export PNG from graphicsPerkota
            graphics_perkota = tab.findChild(QGraphicsView, f"graphics_{value}")
            if graphics_perkota and graphics_perkota.scene() and graphics_perkota.scene().items():
                item = graphics_perkota.scene().items()[0]
                if isinstance(item, QGraphicsPixmapItem):
                    pixmap = item.pixmap()
                    png_path = os.path.join(city_folder, f"{value}_LandCover_Chart_{timestamp}.png")
                    pixmap.save(png_path, "PNG")
                    dialog.logWidget.append(f"Saved PNG for {value} to {png_path}")
                else:
                    dialog.logWidget.append(f"No valid pixmap item in graphics_{value}")
            else:
                dialog.logWidget.append(f"graphics_{value} not populated or found")
            step += 1
            dialog.progressBar.setValue(int((step / total_steps) * 100))

            # Export CSV from table_{value}
            table_perkota = tab.findChild(QTableView, f"table_{value}")
            if table_perkota and table_perkota.model():
                model = table_perkota.model()
                if model.rowCount() == 0 or model.columnCount() == 0:
                    dialog.logWidget.append(f"Table for {value} is empty - skipping CSV export.")
                else:
                    data = []
                    headers = [model.headerData(i, Qt.Horizontal, Qt.DisplayRole) for i in range(model.columnCount())]
                    for row in range(model.rowCount()):
                        row_data = [model.data(model.index(row, col), Qt.DisplayRole) for col in range(model.columnCount())]
                        data.append(row_data)
                    df = pd.DataFrame(data, columns=headers)
                    csv_path = os.path.join(city_folder, f"{value}_Table_{timestamp}.csv")
                    df.to_csv(csv_path, index=False)
                    dialog.logWidget.append(f"Saved CSV for {value} to {csv_path}")
            else:
                dialog.logWidget.append(f"No table or model found for {value} - skipping CSV export.")
            step += 1
            dialog.progressBar.setValue(int((step / total_steps) * 100))

            # Export shapefiles from graphicsCenter_{value}
            graphics_center = tab.findChild(QgsMapCanvas, f"graphicsCenter_{value}")
            if graphics_center:
                layers = graphics_center.layers()
                if not layers:
                    dialog.logWidget.append(f"No layers in graphicsCenter_{value}")
                for layer in layers:
                    layer_name = layer.name()
                    shp_type = "Centroids" if "Centroids" in layer_name else "Centroid_Shifts" if "Centroid_Shifts" in layer_name else None
                    if shp_type:
                        shp_path = os.path.join(city_folder, f"{value}_{shp_type}_{timestamp}.shp")
                        QgsVectorFileWriter.writeAsVectorFormat(
                            layer, shp_path, "UTF-8", layer.crs(), "ESRI Shapefile"
                        )
                        dialog.logWidget.append(f"Saved shapefile for {value} ({shp_type}) to {shp_path}")
            else:
                dialog.logWidget.append(f"graphicsCenter_{value} not found")
            step += 1
            dialog.progressBar.setValue(int((step / total_steps) * 100))

        # Step 2: Export for summary tab
        tab_sum = dialog.tabWidgetAbangku.findChild(QWidget, "tabSum")
        if tab_sum:
            # Create subfolder for global data
            global_folder = os.path.join(base_folder, "Global")
            os.makedirs(global_folder, exist_ok=True)
            dialog.logWidget.append(f"Created subfolder: {global_folder}")

            # Export PNG from graphicsSum
            if dialog.graphicsSum and dialog.graphicsSum.scene() and dialog.graphicsSum.scene().items():
                item = dialog.graphicsSum.scene().items()[0]
                if isinstance(item, QGraphicsPixmapItem):
                    pixmap = item.pixmap()
                    png_path = os.path.join(global_folder, f"Global_LandCover_Chart_{timestamp}.png")
                    pixmap.save(png_path, "PNG")
                    dialog.logWidget.append(f"Saved global PNG to {png_path}")
            step += 1
            dialog.progressBar.setValue(int((step / total_steps) * 100))

            # Export CSV from tableSum
            if dialog.tableSum and dialog.tableSum.model():
                model = dialog.tableSum.model()
                if model.rowCount() == 0 or model.columnCount() == 0:
                    dialog.logWidget.append("Global table (tableSum) is empty - skipping CSV export.")
                else:
                    data = []
                    headers = [model.headerData(i, Qt.Horizontal, Qt.DisplayRole) for i in range(model.columnCount())]
                    for row in range(model.rowCount()):
                        row_data = [model.data(model.index(row, col), Qt.DisplayRole) for col in range(model.columnCount())]
                        data.append(row_data)
                    df = pd.DataFrame(data, columns=headers)
                    csv_path = os.path.join(global_folder, f"Global_Table_{timestamp}.csv")
                    df.to_csv(csv_path, index=False)
                    dialog.logWidget.append(f"Saved global CSV to {csv_path}")
            step += 1
            dialog.progressBar.setValue(int((step / total_steps) * 100))

            # Export shapefiles from graphicsCenterSum
            if dialog.graphicsCenterSum:
                layers = dialog.graphicsCenterSum.layers()
                for layer in layers:
                    layer_name = layer.name()
                    shp_type = "Centroids" if "Centroids" in layer_name else "Centroid_Shifts" if "Centroid_Shifts" in layer_name else None
                    if shp_type:
                        shp_path = os.path.join(global_folder, f"Global_{shp_type}_{timestamp}.shp")
                        QgsVectorFileWriter.writeAsVectorFormat(
                            layer, shp_path, "UTF-8", layer.crs(), "ESRI Shapefile"
                        )
                        dialog.logWidget.append(f"Saved global shapefile ({shp_type}) to {shp_path}")
            step += 1
            dialog.progressBar.setValue(int((step / total_steps) * 100))

        dialog.logWidget.append("Dependent data export completed successfully.")
        dialog.progressBar.setValue(100)
    except Exception as e:
        dialog.logWidget.append(f"Error during export: {str(e)}")
        dialog.progressBar.setValue(0)