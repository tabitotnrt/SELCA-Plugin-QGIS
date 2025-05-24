import os
from qgis.PyQt.QtCore import QAbstractTableModel, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QTableView, QGraphicsScene, QSizePolicy, QGraphicsPixmapItem, QHeaderView
from qgis.core import QgsRasterLayer, QgsTask, QgsApplication, QgsPalettedRasterRenderer, QgsProject
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for QGIS
import matplotlib.pyplot as plt
from osgeo import gdal
import tempfile
from qgis.PyQt.QtGui import QPixmap , QColor
from .assignClass import get_unique_classes
from datetime import datetime  # Add datetime import for timestamp


# TableModel for displaying data in QTableView
class TableModel(QAbstractTableModel):
    def __init__(self, data, headers=None):
        super(TableModel, self).__init__()
        self._data = data
        self._headers = headers

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._data[0])

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and self._headers:
            if orientation == Qt.Horizontal:
                if section < len(self._headers):
                    return self._headers[section]
            if orientation == Qt.Vertical:
                return str(section + 1)
        return super().headerData(section, orientation, role)


def create_change_raster(initial_path, final_path, output_path):
    """Membuat raster yang merepresentasikan transisi tutupan lahan, dengan piksel tidak berubah bernilai -9999 dan penanganan no-data yang tepat."""
    # Buka raster input
    ds1 = gdal.Open(initial_path)
    ds2 = gdal.Open(final_path)
    if ds1 is None or ds2 is None:
        raise ValueError("Tidak dapat membuka file raster.")
    if ds1.RasterXSize != ds2.RasterXSize or ds1.RasterYSize != ds2.RasterYSize:
        raise ValueError("Dimensi raster harus sama.")

    # Ambil band dari raster
    band1 = ds1.GetRasterBand(1)
    band2 = ds2.GetRasterBand(1)

    # Ambil nilai no-data dari metadata raster, gunakan 0 jika tidak ada
    no_data_value1 = band1.GetNoDataValue() if band1.GetNoDataValue() is not None else 0
    no_data_value2 = band2.GetNoDataValue() if band2.GetNoDataValue() is not None else 0

    # Baca array dengan tipe data int32 untuk menghindari overflow
    arr1 = band1.ReadAsArray().astype(np.int32)
    arr2 = band2.ReadAsArray().astype(np.int32)

    # Masker untuk piksel valid (bukan no-data di kedua raster)
    valid_mask = (arr1 != no_data_value1) & (arr2 != no_data_value2)

    # Inisialisasi array output dengan nilai no-data
    output_no_data = 0  # Nilai no-data untuk output
    change_arr = np.full_like(arr1, output_no_data, dtype=np.int32)

    # Tetapkan nilai untuk piksel yang tidak berubah
    unchanged_value = -9999
    change_arr[valid_mask & (arr1 == arr2)] = unchanged_value

    # Tetapkan nilai transisi untuk piksel yang berubah
    change_arr[valid_mask & (arr1 != arr2)] = (arr1 * 100 + arr2)[valid_mask & (arr1 != arr2)]

    # Buat raster output
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(output_path, ds1.RasterXSize, ds1.RasterYSize, 1, gdal.GDT_Int32)
    ds_out.SetGeoTransform(ds1.GetGeoTransform())
    ds_out.SetProjection(ds1.GetProjection())
    band_out = ds_out.GetRasterBand(1)
    band_out.SetNoDataValue(output_no_data)  # Tetapkan nilai no-data pada output
    band_out.WriteArray(change_arr)
    ds_out.FlushCache()
    ds_out = None
# Background task for land cover analysis
class LandCoverAnalysisTask(QgsTask):
    """Background task for computing land cover statistics and transition matrix."""
    resultsComputed = pyqtSignal(dict, dict, list, dict)  # initial_stats, final_stats, additional_stats, transition_matrix
    taskFailed = pyqtSignal(str)

    def __init__(self, initial_path, final_path, additional_paths):
        super().__init__('Land Cover Analysis', QgsTask.CanCancel)
        self.initial_path = initial_path
        self.final_path = final_path
        self.additional_paths = additional_paths
        self.initial_stats = None
        self.final_stats = None
        self.additional_stats = []
        self.transition_matrix = None
        self.exception = None

    def run(self):
        """Perform computations in the background."""
        try:
            self.setProgress(0)
            # Compute area statistics for initial raster
            self.initial_stats = compute_area_statistics(self.initial_path)
            if self.isCanceled():
                return False
            self.setProgress(20)

            # Compute area statistics for additional rasters
            for path in self.additional_paths:
                stats = compute_area_statistics(path)
                self.additional_stats.append(stats)
                if self.isCanceled():
                    return False
            self.setProgress(40)

            # Compute area statistics for final raster
            self.final_stats = compute_area_statistics(self.final_path)
            if self.isCanceled():
                return False
            self.setProgress(50)

            # Compute transition matrix
            self.transition_matrix = compute_transition_matrix(self.initial_path, self.final_path)
            if self.isCanceled():
                return False
            self.setProgress(80)

            self.setProgress(100)
            return True
        except Exception as e:
            self.exception = e
            return False

    def finished(self, result):
        """Handle task completion in the main thread."""
        if result:
            self.resultsComputed.emit(self.initial_stats, self.final_stats, self.additional_stats, self.transition_matrix)
        else:
            self.taskFailed.emit(str(self.exception) if self.exception else "Task was canceled.")

class LandCoverChangeTask(QgsTask):
    """Background task for creating a land cover change raster."""
    changeRasterCreated = pyqtSignal(str)
    taskFailed = pyqtSignal(str)

    def __init__(self, initial_path, final_path):
        super().__init__('Land Cover Change Raster Creation', QgsTask.CanCancel)
        self.initial_path = initial_path
        self.final_path = final_path
        self.output_path = None
        self.exception = None

    def run(self):
        """Perform raster creation in the background with a timestamped file name."""
        try:
            self.setProgress(0)
            # Generate a timestamped file name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Format: YYYYMMDD_HHMMSS
            self.output_path = os.path.join(tempfile.gettempdir(), f'change_raster_{timestamp}.tif')
            create_change_raster(self.initial_path, self.final_path, self.output_path)
            if self.isCanceled():
                return False
            self.setProgress(100)
            return True
        except Exception as e:
            self.exception = e
            return False

    def finished(self, result):
        if result:
            self.changeRasterCreated.emit(self.output_path)
        else:
            self.taskFailed.emit(str(self.exception) if self.exception else "Task was canceled.")

def compute_area_statistics(raster_path):
    """Compute area statistics for each class in a raster, excluding no-data values."""
    # Open the raster file
    ds = gdal.Open(raster_path)
    if ds is None:
        raise ValueError(f"Could not open raster: {raster_path}")

    # Get the first band and its no-data value
    band = ds.GetRasterBand(1)
    no_data_value = band.GetNoDataValue()
    if no_data_value is None:
        no_data_value = 0  # Fallback to 0 if no metadata is available

    # Read raster data and compute pixel area
    raster_array = band.ReadAsArray()
    transform = ds.GetGeoTransform()
    pixel_area = abs(transform[1] * transform[5]) / 1e6  # Convert to km²

    # Get unique classes and their counts, excluding no-data value
    unique_classes, counts = np.unique(raster_array, return_counts=True)
    mask = unique_classes != no_data_value
    unique_classes = unique_classes[mask]
    counts = counts[mask]

    # Calculate area for each class
    area_stats = {int(cls): count * pixel_area for cls, count in zip(unique_classes, counts)}
    return area_stats


def compute_transition_matrix(initial_path, final_path):
    """Compute a transition matrix between two rasters, excluding no-data values."""

    # Open both raster files
    ds1 = gdal.Open(initial_path)
    ds2 = gdal.Open(final_path)
    if ds1 is None or ds2 is None:
        raise ValueError("Could not open one or both raster files.")
    if ds1.RasterXSize != ds2.RasterXSize or ds1.RasterYSize != ds2.RasterYSize:
        raise ValueError("Initial and final rasters must have the same dimensions.")

    # Get bands and no-data values
    band1 = ds1.GetRasterBand(1)
    band2 = ds2.GetRasterBand(1)
    no_data_value1 = band1.GetNoDataValue() if band1.GetNoDataValue() is not None else 0
    no_data_value2 = band2.GetNoDataValue() if band2.GetNoDataValue() is not None else 0

    # Read raster arrays
    arr1 = band1.ReadAsArray().astype(np.int32)
    arr2 = band2.ReadAsArray().astype(np.int32)

    # Create mask to exclude no-data values
    valid_mask = (arr1 != no_data_value1) & (arr2 != no_data_value2)
    arr1_valid = arr1[valid_mask]
    arr2_valid = arr2[valid_mask]

    # Get unique valid classes
    unique_classes = np.union1d(np.unique(arr1_valid), np.unique(arr2_valid))

    # Compute transition counts
    idx1 = np.searchsorted(unique_classes, arr1_valid)
    idx2 = np.searchsorted(unique_classes, arr2_valid)
    counts = np.zeros((len(unique_classes), len(unique_classes)), dtype=np.int64)
    np.add.at(counts, (idx1, idx2), 1)

    # Convert counts to area
    transform = ds1.GetGeoTransform()
    pixel_area = abs(transform[1] * transform[5]) / 1e6  # km²
    counts = counts * pixel_area

    # Build transition matrix dictionary
    transition_matrix = {int(cls): {} for cls in unique_classes}
    for i, cls in enumerate(unique_classes):
        for j, cls2 in enumerate(unique_classes):
            transition_matrix[int(cls)][int(cls2)] = counts[i, j]

    return transition_matrix

# UI update functions (unchanged except for progress tweaks)
def update_graphics_luas(dialog, initial_stats, final_stats, additional_stats):
    """Generate and display a bar chart of area statistics."""
    dialog.progressBar.setValue(50)
    dialog.logWidget.append("Updating graphics Luas...")
    classes = sorted(set(initial_stats.keys()).union(final_stats.keys()))
    initial_areas = [initial_stats.get(cls, 0) for cls in classes]
    final_areas = [final_stats.get(cls, 0) for cls in classes]
    additional_areas = [[stats.get(cls, 0) for cls in classes] for stats in additional_stats]
    
    # Create figure with 2:1 aspect ratio and higher DPI
    plt.rcParams['figure.dpi'] = 100  # Set default DPI
    fig = plt.figure(figsize=(8, 4))  # Reduced size from 12,6 to 8,4
    ax = fig.add_subplot(111)
    
    total_bars = 2 + len(additional_areas)
    width = 0.8 / total_bars
    x = np.arange(len(classes))
    
    positions = [-width / 2, width / 2] if total_bars == 2 else np.linspace(-0.4, 0.4, total_bars)
    
    ax.bar(x + positions[0], initial_areas, width, label=dialog.initialAlias.text() if dialog.initialAlias else 'Initial')
    for i, areas in enumerate(additional_areas):
        alias = dialog.aliasLineEdits[i].text() if i < len(dialog.aliasLineEdits) else f'LC_{i + 2}'
        ax.bar(x + positions[i + 1], areas, width, label=alias)
    ax.bar(x + positions[-1], final_areas, width, label=dialog.finalAlias.text() if dialog.finalAlias else 'Final')
    
    class_names = [dialog.classAliases.get(cls, str(cls)) for cls in classes]
    ax.set_xlabel("Class")
    ax.set_ylabel("Area (km²)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha='right')  # Tilt x-axis labels
    ax.legend()
    
    # Adjust font sizes for better readability
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    
    # Ensure the aspect ratio is maintained when saving
    plt.tight_layout()
    
    if dialog.graphicsLuas.scene() is None:
        dialog.graphicsLuas.setScene(QGraphicsScene())
    dialog.graphicsLuas.scene().clear()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        # Save with optimal DPI for display
        fig.savefig(tmpfile.name, 
                   bbox_inches='tight',
                   dpi=120,  # Adjusted DPI
                   format='png',
                   pad_inches=0.1)
        pixmap = QPixmap(tmpfile.name)
        item = QGraphicsPixmapItem(pixmap)
        item.setTransformationMode(Qt.SmoothTransformation)
        dialog.graphicsLuas.scene().addItem(item)
        dialog.graphicsLuas.fitInView(item, Qt.KeepAspectRatio)
        dialog.temp_files.append(tmpfile.name)
    
    plt.close(fig)
    dialog.progressBar.setValue(70)
    dialog.logWidget.append("Graphics Luas updated.")

def update_table_transition(dialog, transition_matrix):
    """Update the transition matrix table."""
    dialog.progressBar.setValue(80)
    dialog.logWidget.append("Updating table Transition...")
    classes = sorted(transition_matrix.keys())
    data = [[f"{transition_matrix[row_cls][col_cls]:,.2f}" for col_cls in classes] for row_cls in classes]
    
    model = TableModel(data, classes)
    dialog.tableTransition.setModel(model)
    dialog.tableTransition.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    dialog.tableTransition.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
    for i, cls in enumerate(classes):
        model.setHeaderData(i, Qt.Horizontal, str(cls))
        model.setHeaderData(i, Qt.Vertical, str(cls))
    
    dialog.progressBar.setValue(90)
    dialog.logWidget.append("Table Transition updated.")

def update_graphics_transition(dialog, transition_matrix):
    """Generate and display a transition heatmap."""
    dialog.progressBar.setValue(95)
    dialog.logWidget.append("Updating graphics Transition...")
    classes = sorted(transition_matrix.keys())
    matrix = np.zeros((len(classes), len(classes)))
    for i, row_cls in enumerate(classes):
        for j, col_cls in enumerate(classes):
            matrix[i, j] = transition_matrix[row_cls][col_cls]
    
    percentage_matrix = np.zeros_like(matrix)
    for i in range(len(classes)):
        row_total = np.sum(matrix[i, :])
        if row_total > 0:
            percentage_matrix[i] = (matrix[i] / row_total) * 100
    
    fig, ax = plt.subplots()
    cax = ax.matshow(percentage_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{percentage_matrix[i, j]:.2f}", va='center', ha='center', color='black')
    
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    if dialog.graphicsTransition.scene() is None:
        dialog.graphicsTransition.setScene(QGraphicsScene())
    dialog.graphicsTransition.scene().clear()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        fig.savefig(tmpfile.name)
        pixmap = QPixmap(tmpfile.name)
        item = QGraphicsPixmapItem(pixmap)
        item.setTransformationMode(Qt.SmoothTransformation)
        dialog.graphicsTransition.scene().addItem(item)
        dialog.graphicsTransition.fitInView(item, Qt.KeepAspectRatio)
        dialog.temp_files.append(tmpfile.name)
    
    plt.close(fig)
    dialog.progressBar.setValue(100)
    dialog.logWidget.append("Graphics Transition updated.")

def generate_descriptive_text(initial_stats, final_stats, transition_matrix, class_aliases):
    """Generate descriptive text for tableTransition and graphicsLuas."""
    text = []

    # Add description for graphicsLuas
    text.append("**Area Statistics**")
    text.append("This chart compares the area of each land cover class between the initial and final states.")
    text.append("It provides insights into the changes in land cover over time.")
    text.append("For example, an increase in urban areas or a decrease in forested areas can be observed.")

    # Add description for tableTransition
    text.append("\n**Land Cover Transition Matrix**")
    text.append("This table shows the transition of land cover classes from the initial state to the final state.")
    text.append("Each cell represents the area (in km²) that transitioned from one class (row) to another class (column).")
    text.append("The diagonal cells represent areas that remained unchanged.")

    # Add summary of total areas
    total_initial_area = sum(initial_stats.values())
    total_final_area = sum(final_stats.values())
    text.append(f"\n**Summary**")
    text.append(f"Total Initial Area: {total_initial_area:.2f} km²")
    text.append(f"Total Final Area: {total_final_area:.2f} km²")
    text.append("The total area should remain consistent unless there are no-data values or excluded regions.")

    # Calculate total transition area for each class (excluding transitions to itself)
    transition_sums = {
        cls: sum(area for target_cls, area in transitions.items() if target_cls != cls)
        for cls, transitions in transition_matrix.items()
    }
    sorted_transitions = sorted(transition_sums.items(), key=lambda x: x[1], reverse=True)

    # Add most changed classes
    text.append("\n**Most Changed Classes**")
    for cls, total_change in sorted_transitions[:2]:
        alias = class_aliases.get(cls, f"Class {cls}")
        text.append(f"{alias}: Total transition area of {total_change:.2f} km²")

        # Find the two most transitioned-to classes (excluding transitions to itself)
        if cls in transition_matrix:
            transitions = transition_matrix[cls]
            sorted_transitions_to = sorted(
                [(target_cls, area) for target_cls, area in transitions.items() if target_cls != cls],
                key=lambda x: x[1],
                reverse=True
            )
            for target_cls, area in sorted_transitions_to[:2]:
                target_alias = class_aliases.get(target_cls, f"Class {target_cls}")
                text.append(f"  - Transitioned to {target_alias}: {area:.2f} km²")

    # Add least changed classes
    text.append("\n**Least Changed Classes**")
    for cls, total_change in sorted_transitions[-2:]:
        alias = class_aliases.get(cls, f"Class {cls}")
        text.append(f"{alias}: Total transition area of {total_change:.2f} km²")

    return "\n".join(text)

def start_change_map_task(dialog):
    """Initiate the background task for creating the land cover change raster."""
    initialLC = dialog.initialLC.currentLayer()
    finalLC = dialog.finalLC.currentLayer()

    if not (isinstance(initialLC, QgsRasterLayer) and isinstance(finalLC, QgsRasterLayer)):
        dialog.logWidget.append("Error: Initial and final layers must be valid raster layers.")
        return

    initial_path = initialLC.source()
    final_path = finalLC.source()

    dialog.change_task = LandCoverChangeTask(initial_path, final_path)
    dialog.change_task.progressChanged.connect(lambda: update_progress_change(dialog))
    dialog.change_task.changeRasterCreated.connect(lambda raster_path: on_change_raster_created(dialog, raster_path))
    dialog.change_task.taskFailed.connect(lambda error_message: on_change_task_failed(dialog, error_message))

    QgsApplication.taskManager().addTask(dialog.change_task)
    dialog.logWidget.append("Starting change map creation...")

def update_progress_change(dialog):
    """Update the progress bar during change task execution."""
    task = dialog.change_task  # Sender is not directly available; use stored task
    dialog.progressBar.setValue(int(task.progress()))

def on_change_raster_created(dialog, raster_path):
    """Handle successful creation of the change raster with 'Unchanged' class."""
    dialog.logWidget.append("Change raster created. Loading into QGIS...")
    try:
        change_layer = QgsRasterLayer(raster_path, 'Land Cover Change')
        if not change_layer.isValid():
            raise ValueError("Could not load change raster.")
        QgsProject.instance().addMapLayer(change_layer)

        # Set transparency for no-data value (0)
        change_layer.dataProvider().setNoDataValue(1, 0)  # Band 1, no-data = 0
        change_layer.triggerRepaint()

        initial_path = dialog.initialLC.currentLayer().source()
        unique_classes = sorted(set(get_unique_classes(initial_path)).union(get_unique_classes(dialog.finalLC.currentLayer().source())))
        classes = []

        # Add "Unchanged" class for value -9999
        classes.append(QgsPalettedRasterRenderer.Class(-9999, QColor('gray'), "Unchanged"))

        # Add transition classes for changed pixels
        for initial_cls in unique_classes:
            for final_cls in unique_classes:
                if initial_cls != final_cls:  # Skip unchanged cases
                    value = initial_cls * 100 + final_cls
                    initial_label = dialog.classAliases.get(initial_cls, str(initial_cls))
                    final_label = dialog.classAliases.get(final_cls, str(final_cls))
                    label = f"{initial_label} to {final_label}"
                    color = QColor.fromHslF(np.random.random(), 1, 0.5)
                    classes.append(QgsPalettedRasterRenderer.Class(value, color, label))

        renderer = QgsPalettedRasterRenderer(change_layer.dataProvider(), 1, classes)
        change_layer.setRenderer(renderer)
        change_layer.triggerRepaint()

        dialog.logWidget.append("Change map loaded with symbology (Unchanged class included).")
        dialog.progressBar.setValue(100)
    except Exception as e:
        dialog.logWidget.append(f"Error loading change raster: {str(e)}")
        dialog.progressBar.setValue(0)

def on_change_task_failed(dialog, error_message):
    """Handle task failure or cancellation."""
    dialog.logWidget.append(f"Change map creation failed: {error_message}")
    dialog.progressBar.setValue(0)