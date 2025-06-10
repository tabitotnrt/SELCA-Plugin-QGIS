from .crosstab import compute_area_statistics, compute_transition_matrix, TableModel
import numpy as np
import matplotlib.pyplot as plt
from qgis.PyQt.QtWidgets import QHeaderView, QGraphicsScene, QGraphicsPixmapItem
from qgis.PyQt.QtCore import Qt
from .crosstab import TableModel
import tempfile
from qgis.PyQt.QtGui import QPixmap

def perform_forecasting(dialog):
    """Perform estimating based on initial area and transition matrix using iterative method with area conservation."""
    try:
        # Fetch temporal jump multiplier
        temporal_jump = dialog.temporalJump.value()
        dialog.logWidget.append(f"Temporal jump multiplier: {temporal_jump}")

        # Fetch data from tableInitialArea
        initial_model = dialog.tableInitialArea.model()
        if not initial_model:
            dialog.logWidget.append("Error: Initial area data is missing.")
            return

        initial_data = {}
        for row in range(initial_model.rowCount(None)):
            class_code = int(initial_model.data(initial_model.index(row, 0), Qt.DisplayRole))  # Kolom 0: kode kelas
            area = float(initial_model.data(initial_model.index(row, 2), Qt.DisplayRole))      # Kolom 2: area
            initial_data[class_code] = area

        dialog.logWidget.append(f"Initial areas: {initial_data}")

        # Fetch data from tableTransitionMat
        transition_model = dialog.tableTransitionMat.model()
        if not transition_model:
            dialog.logWidget.append("Error: Transition matrix data is missing.")
            return

        classes = [int(transition_model.headerData(col, Qt.Horizontal, Qt.DisplayRole)) for col in range(transition_model.columnCount(None))]
        transition_matrix = np.zeros((len(classes), len(classes)))

        for row in range(transition_model.rowCount(None)):
            for col in range(transition_model.columnCount(None)):
                value = transition_model.data(transition_model.index(row, col), Qt.DisplayRole).replace('%', '')
                transition_matrix[row, col] = float(value) / 100.0

        dialog.logWidget.append(f"Transition matrix before normalization:\n{transition_matrix}")

        # Normalize transition matrix rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)
        dialog.logWidget.append(f"Transition matrix after normalization:\n{transition_matrix}")

        # Prepare initial area vector
        initial_areas = np.array([initial_data.get(cls, 0) for cls in classes])
        dialog.logWidget.append(f"Initial areas array: {initial_areas}")
        total_area = np.sum(initial_areas)

        # Iterative forecasting with area conservation
        forecasted_areas = initial_areas.copy()
        for step in range(temporal_jump):
            forecasted_areas = forecasted_areas.dot(transition_matrix)
            forecasted_areas = (forecasted_areas / np.sum(forecasted_areas)) * total_area  # Preserve total area

        dialog.logWidget.append(f"Forecasted areas array after {temporal_jump} steps: {forecasted_areas}")

        # Update tableForecastedArea
        total_forecasted_area = np.sum(forecasted_areas)
        forecasted_data = [
            [dialog.classAliases.get(cls, str(cls)), f"{area:.2f}", f"{(area / total_forecasted_area) * 100:.2f}%"]
            for cls, area in zip(classes, forecasted_areas)
        ]
        headers = ["Class Name", "Estimated Area (km²)", "Percentage"]
        forecasted_model = TableModel(forecasted_data, headers)
        dialog.tableForecastedArea.setModel(forecasted_model)
        dialog.tableForecastedArea.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        dialog.tableForecastedArea.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Update graphicsInitialForecast
        dialog.logWidget.append("Generating comparison graph...")
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(classes))
        width = 0.4

        ax.bar(x - width / 2, initial_areas, width, label="Initial Area")
        ax.bar(x + width / 2, forecasted_areas, width, label=f"Forecasted Area ({temporal_jump} steps)")

        ax.set_xlabel("Class")
        ax.set_ylabel("Area (km²)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(cls) for cls in classes], rotation=15, ha='right')
        ax.legend()

        plt.tight_layout()

        if dialog.graphicsInitialForecast.scene() is None:
            dialog.graphicsInitialForecast.setScene(QGraphicsScene())
        dialog.graphicsInitialForecast.scene().clear()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight', dpi=120, format='png', pad_inches=0.1)
            pixmap = QPixmap(tmpfile.name)
            item = QGraphicsPixmapItem(pixmap)
            item.setTransformationMode(Qt.SmoothTransformation)
            dialog.graphicsInitialForecast.scene().addItem(item)
            dialog.graphicsInitialForecast.fitInView(item, Qt.KeepAspectRatio)
            dialog.temp_files.append(tmpfile.name)

        plt.close(fig)
        dialog.logWidget.append("Generating completed successfully.")
    except Exception as e:
        dialog.logWidget.append(f"Error during estimating: {str(e)}")


def fetch_forecast_data(dialog):
    """Fetch data for initial area and transition matrix."""
    finalLC = dialog.finalLC.currentLayer()

    if not finalLC:
        dialog.logWidget.append("Error: Final land cover layer is not selected.")
        return

    try:
        # Populate tableInitialArea
        dialog.logWidget.append("Generating initial area data...")
        area_stats = compute_area_statistics(finalLC.source())
        total_area = sum(area_stats.values())

        # Tampilkan Class Code dan Class Name
        data = [
            [cls, dialog.classAliases.get(cls, str(cls)), f"{area:.2f}", f"{(area / total_area) * 100:.2f}%"]
            for cls, area in sorted(area_stats.items())
        ]
        headers = ["Class Code", "Class Name", "Area (km²)", "Percentage"]
        model = TableModel(data, headers)
        dialog.tableInitialArea.setModel(model)
        dialog.tableInitialArea.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        dialog.tableInitialArea.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Populate tableTransitionMat
        dialog.logWidget.append("Fetching transition matrix data...")
        transition_matrix = compute_transition_matrix(dialog.initialLC.currentLayer().source(), finalLC.source())
        classes = sorted(transition_matrix.keys())
        matrix = [[transition_matrix[row_cls].get(col_cls, 0) for col_cls in classes] for row_cls in classes]

        # Convert to percentage
        percentage_matrix = []
        for row in matrix:
            row_total = sum(row)
            percentage_matrix.append([f"{(value / row_total) * 100:.2f}" if row_total > 0 else "0.00" for value in row])

        model = TableModel(percentage_matrix, classes)
        dialog.tableTransitionMat.setModel(model)
        dialog.tableTransitionMat.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        dialog.tableTransitionMat.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, cls in enumerate(classes):
            model.setHeaderData(i, Qt.Horizontal, str(cls))
            model.setHeaderData(i, Qt.Vertical, str(cls))

        dialog.logWidget.append("Data fetched successfully.")
    except Exception as e:
        dialog.logWidget.append(f"Error fetching data: {str(e)}")
