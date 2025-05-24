import os
import inspect
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
from .Meanspatial import MeanSpatial
from .GlobalMoransI import GlobalMoransI
from .LocalMorans import LocalMoransI
from .LQ_algorithm import LQ

class SelcaProvider(QgsProcessingProvider):

    def loadAlgorithms(self):
        # Grup: Urban Sprawl Analysis
        self.addAlgorithm(MeanSpatial())

        # Grup: Spatial Analysis
        self.addAlgorithm(GlobalMoransI())
        self.addAlgorithm(LocalMoransI())
        self.addAlgorithm(LQ())

    def id(self):
        return "SELCA"

    def name(self):
        return self.tr("Socio-Economic and Land Cover Analysis (SELCA)")

    def icon(self):
        frame = inspect.currentframe()
        cmd_folder = os.path.dirname(inspect.getfile(frame)) if frame else os.path.dirname(__file__)
        icon_path = os.path.join(cmd_folder, '..', 'Icons', 'logo1.png')
        return QIcon(icon_path) if os.path.exists(icon_path) else QIcon()

    def longName(self):
        return self.name()
