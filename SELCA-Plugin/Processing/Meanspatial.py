import numpy as np
import rasterio
import math
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsFeatureSink,
    QgsFeature,
    QgsFields,
    QgsField,
    QgsWkbTypes,
    QgsCoordinateReferenceSystem,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingException,
    QgsProcessing,
    QgsProcessingMultiStepFeedback,
    QgsProcessingUtils,
    QgsProcessingContext
)
from PyQt5.QtCore import QVariant, QCoreApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QUrl

import os

class MeanSpatial(QgsProcessingAlgorithm):

    INPUT = 'INPUT'
    BUILD = "BUILD"
    OUTPUT = 'OUTPUT'  # Only one output parameter for points

    def name(self):
        return 'CityCenterTrendAnalysis'

    def displayName(self):
        return self.tr('City Center Trend Analysis')

    def group(self):
        return self.tr('Centroid Shift Analysis')

    def groupId(self):
        return 'CentroidShiftAnalysis'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return MeanSpatial()
    
    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), '..', 'Icons', 'logo4.png')
        if not os.path.exists(icon_path):
            raise FileNotFoundError(f"Icon file not found at {icon_path}")
        return QIcon(icon_path)

    def helpUrl(self):
        file = os.path.dirname(__file__) + '/selcahelpEN.html'
        if not os.path.exists(file):
            return ''
        return QUrl.fromLocalFile(file).toString(QUrl.FullyEncoded)

    def shortHelpString(self):
        return (
            "Algorithm Overview:\n"
            "This tool calculates the 'center' of built-up areas on a series of land cover maps and tracks "
            "how that center shifts over time. For each map, it determines the average location of built-up areas, "
            "marks it as a point, and draws lines between consecutive points to illustrate the movement in both "
            "distance and direction.\n\n"
            "Input Parameters:\n"
            " - Land Cover Classification Layers: Provide multiple raster layers representing land cover maps, "
            "   arranged in chronological order from oldest to newest. All layers must share the same coordinate system.\n"
            " - Built-up Area Value: Enter the numeric value that indicates built-up areas within your raster data "
            "(default is typically 1). This value identifies which pixels to consider when calculating the central point."
        )

    def initAlgorithm(self, config=None):
        """Define the inputs and output of the algorithm."""
        # Input: Multiple raster layers
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT,
                self.tr("Input Land Cover Classification (Oldest to Newest)"),
                layerType=QgsProcessing.TypeRaster,
                optional=False
            )
        )

        # Input: Value representing built-up areas
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUILD,
                self.tr("Input Built-up area value"),
                QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                optional=False
            )
        )

        # Output: Point layer for centroids (primary output)
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output Points')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Main processing logic."""
        # Get inputs
        input_rasters = self.parameterAsLayerList(parameters, self.INPUT, context)
        built_up_value = self.parameterAsInt(parameters, self.BUILD, context)

        # Validate input
        if not input_rasters:
            raise QgsProcessingException("No raster layers provided!")

        # Ensure consistent CRS
        first_crs = input_rasters[0].crs()
        for raster in input_rasters:
            if raster.crs() != first_crs:
                raise QgsProcessingException(f"Projection mismatch: {raster.name()} is not in {first_crs.authid()}!")

        # Define fields for point output with X and Y coordinates
        point_fields = QgsFields()
        point_fields.append(QgsField("Layer_Name", QVariant.String))
        point_fields.append(QgsField("X", QVariant.Double))
        point_fields.append(QgsField("Y", QVariant.Double))

        # Create point sink (primary output)
        (point_sink, point_dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            point_fields, QgsWkbTypes.Point, first_crs
        )

        # Store points and layer names
        points_with_names = []

        # Set up progress feedback
        total_steps = len(input_rasters) + 1  # +1 for line creation if applicable
        multi_feedback = QgsProcessingMultiStepFeedback(total_steps, feedback)

        # Process each raster to calculate centroids
        for i, raster_layer in enumerate(input_rasters):
            multi_feedback.setCurrentStep(i)
            if multi_feedback.isCanceled():
                return {}

            layer_name = raster_layer.name()
            raster_path = raster_layer.source()

            multi_feedback.pushInfo(f"Processing {layer_name} (Step {i + 1}/{total_steps})")

            # Read raster data
            with rasterio.open(raster_path) as src:
                data = src.read(1)  # Assuming single-band raster
                transform = src.transform

                # Find built-up area pixels
                rows, cols = np.where(data == built_up_value)
                if len(rows) == 0:
                    multi_feedback.pushWarning(f"No built-up areas in {layer_name}")
                    continue

                # Calculate centroid
                mean_row, mean_col = np.mean(rows), np.mean(cols)
                centroid_x, centroid_y = transform * (mean_col, mean_row)
                centroid_x = float(centroid_x)
                centroid_y = float(centroid_y)
                point = QgsPointXY(centroid_x, centroid_y)

                # Store point with layer name
                points_with_names.append((layer_name, point))

                # Add point feature with X, Y coordinates
                point_feature = QgsFeature()
                point_feature.setAttributes([layer_name, centroid_x, centroid_y])
                point_feature.setGeometry(QgsGeometry.fromPointXY(point))
                point_sink.addFeature(point_feature, QgsFeatureSink.FastInsert)

        # Create separate line features if multiple points exist
        if len(points_with_names) > 1:
            multi_feedback.setCurrentStep(total_steps - 1)
            if multi_feedback.isCanceled():
                return {}

            multi_feedback.pushInfo(f"Creating movement lines (Step {total_steps}/{total_steps})")

            # Define fields for line output with Distance and Direction
            line_fields = QgsFields()
            line_fields.append(QgsField("Connection", QVariant.String))
            line_fields.append(QgsField("Azimuth", QVariant.Double))
            line_fields.append(QgsField("Distance", QVariant.Double))
            line_fields.append(QgsField("Direction", QVariant.String))

            # Create a temporary line sink
            line_dest_id = QgsProcessingUtils.generateTempFilename('lines.shp')
            (line_sink, line_dest_id) = QgsProcessingUtils.createFeatureSink(
                line_dest_id, context,
                line_fields, QgsWkbTypes.LineString, first_crs
            )

            # Generate separate lines between consecutive points
            for i in range(len(points_with_names) - 1):
                layer_name1, point1 = points_with_names[i]
                layer_name2, point2 = points_with_names[i + 1]

                # Create line geometry
                line_geom = QgsGeometry.fromPolylineXY([point1, point2])

                # Calculate azimuth (directional change)
                dx = point2.x() - point1.x()
                dy = point2.y() - point1.y()
                azimuth = (math.degrees(math.atan2(dx, dy)) + 360) % 360

                # Calculate distance
                distance = math.sqrt(dx**2 + dy**2)  # Euclidean distance in CRS units (e.g., meters if CRS is projected)

                # Determine cardinal direction from azimuth
                if 337.5 <= azimuth < 360 or 0 <= azimuth < 22.5:
                    direction = "North"
                elif 22.5 <= azimuth < 67.5:
                    direction = "North-East"
                elif 67.5 <= azimuth < 112.5:
                    direction = "East"
                elif 112.5 <= azimuth < 157.5:
                    direction = "South-East"
                elif 157.5 <= azimuth < 202.5:
                    direction = "South"
                elif 202.5 <= azimuth < 247.5:
                    direction = "South-West"
                elif 247.5 <= azimuth < 292.5:
                    direction = "West"
                elif 292.5 <= azimuth < 337.5:
                    direction = "North-West"

                # Create connection attribute
                connection = f"{layer_name1} - {layer_name2}"

                # Add line feature with all attributes
                line_feature = QgsFeature()
                line_feature.setAttributes([connection, azimuth, distance, direction])
                line_feature.setGeometry(line_geom)
                line_sink.addFeature(line_feature, QgsFeatureSink.FastInsert)

            # Add the line layer to the project
            context.addLayerToLoadOnCompletion(
                line_dest_id,
                QgsProcessingContext.LayerDetails('Movement Lines', context.project(), 'OUTPUT_LINES')
            )

        multi_feedback.pushInfo("Processing complete!")
        return {self.OUTPUT: point_dest_id}