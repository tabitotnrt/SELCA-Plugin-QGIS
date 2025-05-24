# -*- coding: utf-8 -*-
"""
/***************************************************************************
 LocationQuotient
 A QGIS plugin
***************************************************************************/
"""

__author__ = 'Tabito'
__date__ = '2024-10-24'
__copyright__ = '(C) 2024 by Tabito'

import processing
import os
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterString,
    QgsProcessingParameterFileDestination,
    QgsFeatureSink,
    QgsField,
    Qgis,
    QgsSymbol,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
    QgsProcessingUtils,
    QgsProcessingException
)
from PyQt5.QtCore import QVariant
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import QUrl

class LQ(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    VARIABLEX = 'VARIABLEX'
    VARIABLEY = 'VARIABLEY'
    LQFIELD = 'LQFIELD'
    CSV_OUTPUT = 'CSV_OUTPUT'

    def name(self):
        return 'Economic Sector Specialization Analysis'

    def displayName(self):
        return self.tr('Economic Sector Specialization Analysis')

    def group(self):
        return "Spatial Analysis"

    def groupId(self):
        return "Spatial Analysis"

    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), '..', 'Icons', 'logo5.png')
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
            self.INPUT, 'Layer',
            types=[QgsProcessing.TypeVectorAnyGeometry]
        ))
        self.addParameter(QgsProcessingParameterField(
            self.VARIABLEX, 'Variable X (Sector)', type=QgsProcessingParameterField.Numeric,
            parentLayerParameterName=self.INPUT
        ))
        self.addParameter(QgsProcessingParameterField(
            self.VARIABLEY, 'Variable Y (Total)', type=QgsProcessingParameterField.Numeric,
            parentLayerParameterName=self.INPUT
        ))
        self.addParameter(QgsProcessingParameterString(
            self.LQFIELD, 'Name for LQ Field', defaultValue='LQ'
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.CSV_OUTPUT, 'CSV Output File', fileFilter='CSV files (*.csv)'
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, 'Location Quotient Output', createByDefault=True
        ))

    def processAlgorithm(self, parameters, context, model_feedback):
        layer = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        variableX = self.parameterAsString(parameters, self.VARIABLEX, context)
        variableY = self.parameterAsString(parameters, self.VARIABLEY, context)
        lqField = self.parameterAsString(parameters, self.LQFIELD, context)
        csv_path = self.parameterAsFileOutput(parameters, self.CSV_OUTPUT, context)
        self.field = lqField

        if not layer:
            raise QgsProcessingException("Layer not found!")

        field_names = [f.name() for f in layer.fields()]
        if variableX not in field_names or variableY not in field_names:
            raise QgsProcessingException(
                f"Selected field(s) not found in layer.\nAvailable fields: {field_names}"
            )

        pr = layer.dataProvider()

        existing_fields = [field.name() for field in layer.fields()]
        new_fields = []
        if lqField not in existing_fields:
            new_fields.append(QgsField(lqField, QVariant.Double, len=10, prec=5))
        if 'LQ_Class' not in existing_fields:
            new_fields.append(QgsField('LQ_Class', QVariant.String))
        if 'Interpretation' not in existing_fields:
            new_fields.append(QgsField('Interpretation', QVariant.String))

        if new_fields:
            pr.addAttributes(new_fields)
            layer.updateFields()
        else:
            return {'Error': 'LQ or class fields already exist'}

        idx1 = layer.fields().indexFromName(variableX)
        idx2 = layer.fields().indexFromName(variableY)
        idx_lq = layer.fields().indexFromName(lqField)
        idx_class = layer.fields().indexFromName('LQ_Class')
        idx_interp = layer.fields().indexFromName('Interpretation')

        X = sum(ftr[idx1] for ftr in layer.getFeatures() if ftr[idx1] is not None)
        Y = sum(ftr[idx2] for ftr in layer.getFeatures() if ftr[idx2] is not None)

        if Y == 0:
            raise QgsProcessingException("Sum of Y values is zero, cannot compute LQ.")

        XoverY = X / Y

        attr_updates = {}
        for ftr in layer.getFeatures():
            fid = ftr.id()
            if ftr[idx2] == 0 or ftr[idx2] is None or ftr[idx1] is None:
                LQ = None
                cls = None
                interp = None
            else:
                xovery = ftr[idx1] / ftr[idx2]
                LQ = xovery / XoverY
                if LQ > 1:
                    cls = 'Superior'
                    interp = f"The region has a higher specialization in '{variableX}' than average."
                else:
                    cls = 'Not Superior'
                    interp = f"The region has equal or less specialization in '{variableX}' than average."

            attr_updates[fid] = {
                idx_lq: LQ,
                idx_class: cls,
                idx_interp: interp
            }

        pr.changeAttributeValues(attr_updates)
        layer.updateFields()

        (sink, self.dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                                    layer.fields(), layer.wkbType(), layer.sourceCrs())
        for feature in layer.getFeatures():
            sink.addFeature(feature, QgsFeatureSink.FastInsert)

        model_feedback.pushInfo(f"\n📁 CSV file will be saved to:\n{csv_path}")

        model_feedback.pushInfo("\n--- Location Quotient (LQ) Analysis Report ---")
        model_feedback.pushInfo(f"Input sector variable (X): '{variableX}'")
        model_feedback.pushInfo(f"Input total variable (Y): '{variableY}'")
        model_feedback.pushInfo(f"Calculated field: '{lqField}'\n")

        model_feedback.pushInfo("📊 Definition:")
        model_feedback.pushInfo(f"LQ = ( {variableX} / {variableY} ) ÷ ( Σ{variableX} / Σ{variableY} )")

        model_feedback.pushInfo("\n📈 Classification:")
        model_feedback.pushInfo(" - 'Superior'      → LQ > 1 → Above average specialization")
        model_feedback.pushInfo(" - 'Not Superior'  → LQ ≤ 1 → At or below average specialization")

        model_feedback.pushInfo("\n📘 Output Fields:")
        model_feedback.pushInfo(f" - '{lqField}': Numeric Location Quotient value")
        model_feedback.pushInfo(" - 'LQ_Class': Category label ('Superior' or 'Not Superior')")
        model_feedback.pushInfo(" - 'Interpretation': Explanation of result in plain English")

        with open(csv_path, 'w', encoding='utf-8') as file:
            headers = [field.name() for field in layer.fields()]
            file.write(','.join(headers) + '\n')
            for feat in layer.getFeatures():
                values = [str(feat[field.name()]) if feat[field.name()] is not None else '' for field in layer.fields()]
                file.write(','.join(values) + '\n')

        model_feedback.pushInfo(f"\n✅ CSV export completed to:\n{csv_path}")
        model_feedback.pushInfo("✔️ LQ calculation and classification completed.")

        return {self.OUTPUT: self.dest_id}

    def postProcessAlgorithm(self, context, feedback):
        layer = QgsProcessingUtils.mapLayerFromString(self.dest_id, context)
        if not layer:
            return {self.OUTPUT: self.dest_id}

        categories = []

        symbol_superior = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol_superior.setColor(QColor("green"))
        categories.append(QgsRendererCategory("Superior", symbol_superior, "Superior"))

        symbol_notsup = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol_notsup.setColor(QColor("gray"))
        categories.append(QgsRendererCategory("Not Superior", symbol_notsup, "Not Superior"))

        renderer = QgsCategorizedSymbolRenderer('LQ_Class', categories)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        return {self.OUTPUT: self.dest_id}

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def shortHelpString(self):
        return (
            "Performs economic sector specialization analysis using the Location Quotient (LQ) method.\n\n"
            "This plugin, Economic Sector Specialization Analysis, helps identify whether a region "
            "is specialized in a specific economic sector compared to a reference area.\n\n"
            "Inputs:\n"
            "- x: Sector-specific value for the local unit\n"
            "- y: Total economic value for the local unit\n"
            "- X: Sector-specific value for the reference area (e.g., national or provincial total)\n"
            "- Y: Total economic value for the reference area\n\n"
            "Outputs:\n"
            "- LQ value indicating relative specialization\n"
            "- Classification as 'Superior' or 'Not Superior'\n"
            "- Interpretation text per feature\n"
            "- Output vector layer with symbology\n"
            "- CSV export with all results\n\n"
            "Useful for regional economic analysis, identifying sectoral strengths, "
            "and supporting planning and policy decisions."
        )

    def createInstance(self):
        return LQ()
