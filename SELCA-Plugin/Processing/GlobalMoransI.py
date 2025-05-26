# -*- coding: utf-8 -*-
"""
Global Spatial Pattern Analysis Plugin
"""

__author__ = 'Tabito'
__date__ = '2024-10-24'
__copyright__ = '(C) 2024 by Tabito'

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsVectorFileWriter
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QUrl
import os
import tempfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import Queen, Rook, KNN, DistanceBand, spatial_lag
from esda.moran import Moran

class GlobalMoransI(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    VARIABLE = 'VARIABLE'
    METHOD = 'METHOD'
    PARAM = 'PARAM'
    CONFIDENCE = 'CONFIDENCE'
    PLOT_OUTPUT = 'PLOT_OUTPUT'

    def name(self):
        return 'Global Spatial Pattern Analysis'

    def displayName(self):
        return self.tr("Global Spatial Pattern Analysis")

    def group(self):
        return "Spatial Analysis"

    def groupId(self):
        return "Spatial Analysis"

    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), '..', 'Icons', 'logo2.png')
        if not os.path.exists(icon_path):
            raise FileNotFoundError(f"Icon file not found at {icon_path}")
        return QIcon(icon_path)
    
    def helpUrl(self):
        file = os.path.dirname(__file__) + '/selcahelpEN.html'
        if not os.path.exists(file):
            return ''
        return QUrl.fromLocalFile(file).toString(QUrl.FullyEncoded)
    
    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT,
            'Input layer',
            types=[QgsProcessing.TypeVectorPolygon, QgsProcessing.TypeVectorPoint]
        ))
        self.addParameter(QgsProcessingParameterField(
            self.VARIABLE,
            'Variable X',
            type=QgsProcessingParameterField.Numeric,
            parentLayerParameterName=self.INPUT
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.METHOD,
            'Spatial Weight Method',
            options=['Queen contiguity', 'Rook contiguity', 'K Nearest Neighbors', 'Distance Band'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.PARAM,
            type=QgsProcessingParameterNumber.Double,
            description='K (for KNN) or Threshold Distance (for Distance Band)\n(Ignored if using Queen or Rook)',
            defaultValue=1.0,
            minValue=0.1
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.CONFIDENCE,
            'Confidence Level',
            options=['99%', '95%', '90%'],
            defaultValue=1
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.PLOT_OUTPUT,
            'Moran Scatterplot Output (PNG)',
            fileFilter='PNG files (*.png)'
        ))

    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        field = self.parameterAsString(parameters, self.VARIABLE, context)
        method = self.parameterAsInt(parameters, self.METHOD, context)
        param_value = self.parameterAsDouble(parameters, self.PARAM, context)
        conf_level = self.parameterAsInt(parameters, self.CONFIDENCE, context)
        plot_output = self.parameterAsFileOutput(parameters, self.PLOT_OUTPUT, context)

        conf_labels = ['99%', '95%', '90%']
        conf_thresholds = [0.01, 0.05, 0.10]
        conf_text = conf_labels[conf_level]
        p_threshold = conf_thresholds[conf_level]

        # Save to shapefile
        temp = os.path.join(tempfile.gettempdir(), 'global_moran_input.shp')
        QgsVectorFileWriter.writeAsVectorFormat(layer, temp, 'utf-8', layer.crs(), 'ESRI Shapefile')

        gdf = gpd.read_file(temp)

        if layer.geometryType() == 2 and method in (2, 3):
            gdf['geometry'] = gdf.centroid

        gdf = gdf[[field, 'geometry']].dropna()

        if method == 0:
            w = Queen.from_dataframe(gdf)
        elif method == 1:
            w = Rook.from_dataframe(gdf)
        elif method == 2:
            w = KNN.from_dataframe(gdf, k=int(param_value))
        elif method == 3:
            w = DistanceBand.from_dataframe(gdf, threshold=param_value)

        y = gdf[field]
        moran = Moran(y, w)

        MI = moran.I
        EI = moran.EI
        Z = moran.z_norm
        P = moran.p_norm

        mean_val = y.mean()
        std_val = y.std()
        z = (y - mean_val) / std_val
        wz = pd.Series(spatial_lag.lag_spatial(w, z))

        fig, ax = plt.subplots()
        sns.regplot(x=z, y=wz, ci=None, ax=ax)
        ax.axhline(0, color='k', linestyle='--')
        ax.axvline(0, color='k', linestyle='--')
        ax.set_title(f"Global Moran's I Scatterplot\nI = {MI:.4f}, p = {P:.4f}")
        ax.set_xlabel("Standardized Variable (Z)")
        ax.set_ylabel("Spatial Lag (W * Z)")
        plt.tight_layout()
        plt.savefig(plot_output, dpi=150)
        plt.close()

        pattern = 'clustered' if MI > 0 else 'dispersed' if MI < 0 else 'random'
        signif = 'statistically significant' if P < p_threshold else 'not statistically significant'

        explanation = {
            'clustered': "→ The data shows *positive spatial autocorrelation*, meaning similar values (e.g. high with high, low with low) tend to group together.",
            'dispersed': "→ The data shows *negative spatial autocorrelation*, meaning dissimilar values (e.g. high next to low) are more common.",
            'random': "→ The data shows no meaningful spatial pattern; values are randomly distributed across space."
        }

        feedback.pushInfo(f"""
====== Global Spatial Pattern Analysis Summary ======

Layer: {layer.name()}
Variable: {field}
Method: {['Queen', 'Rook', 'KNN', 'Distance Band'][method]}
Confidence Level: {conf_text}

-- Descriptive Stats --
Min: {y.min():.2f}
Max: {y.max():.2f}
Mean: {mean_val:.2f}
Std Dev: {std_val:.2f}

-- Global Moran's I --
Moran's I: {MI:.5f}
Expected I: {EI:.5f}
Z-score: {Z:.5f}
P-value: {P:.5f}

-- Interpretation --
→ Pattern: {pattern}
{explanation[pattern]}
→ Significance: {signif} (p < {p_threshold})
→ Based on {conf_text} confidence level

Lay Explanation:
- Clustered → Similar values are near each other (e.g. high-income areas near other high-income areas)
- Dispersed → Opposite values are near each other (e.g. rich areas next to poor ones)
- Random → No consistent pattern across space

Scatterplot saved to:
{plot_output}
""")

        return {'PLOT_OUTPUT': plot_output}

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def shortHelpString(self):
        return (
            "Analyzes global spatial pattern using Moran's I.\n\n"
            "Features:\n"
            "- Computes Global Moran's I, Expected I, Z-score, and p-value\n"
            "- Selectable spatial weight method: Queen, Rook, KNN, Distance Band\n"
            "- Selectable confidence level (99%, 95%, 90%)\n"
            "- Moran scatterplot output in PNG format\n"
            "- Uses GeoPandas backend for improved geometry handling\n\n"
            "Interpretation:\n"
            "- Clustered → Similar values group together in space\n"
            "- Dispersed → Opposite values are near each other\n"
            "- Random → No spatial structure in the data"
        )

    def createInstance(self):
        return GlobalMoransI()
