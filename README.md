<div style="display: flex; align-items: center;">
  <p align="center"> 
    <img src="/SELCA-Plugin/icon.png" alt="SELCA Logo" style="width: 250px; height: 250px;">
    <h1>Socio-Economic and Land Cover Analysis (SELCA)</h1>
  </p>
</div>

---

**SELCA** is a QGIS plugin designed to analyze the relationship between land cover changes and socio-economic indicators such as population and GDRP. It provides spatial statistical tools, regression models, cluster analysis, and forecasting functionalities to support land use planning, sustainability assessments, and development policy.

---

## 🔧 List of Tools

---

<div style="display: flex; align-items: center;">
  <img src="/icons/assignclass.png" alt="Assign Class" style="width: 100px; height: 100px; margin-right: 20px;">
  <h2>Assign Class</h2>
</div>

Reassigns descriptive labels to land cover raster classes. This helps users translate numeric raster codes into human-readable class names for better visualization and analysis.

---

<div style="display: flex; align-items: center;">
  <img src="/icons/transitionmatrix.png" alt="Transition Matrix" style="width: 100px; height: 100px; margin-right: 20px;">
  <h2>Land Cover Insight</h2>
</div>

Generates land cover transition matrix from two raster timeframes. Includes area change stats, transition heatmaps, and descriptive insights for each class transformation.

| Input | Output |
|-------|--------|
| ![Input](/icons/ToolExample/InputMatrix.png) | ![Output](/icons/ToolExample/OutputMatrix.png) |

---

<div style="display: flex; align-items: center;">
  <img src="/icons/dependentinsight.png" alt="Dependent Insight" style="width: 100px; height: 100px; margin-right: 20px;">
  <h2>Dependent Insight</h2>
</div>

Visualizes correlation between land cover changes and dependent variables like population and GDRP, along with centroid shift mapping.

---

<div style="display: flex; align-items: center;">
  <img src="/icons/regression.png" alt="Regression" style="width: 100px; height: 100px; margin-right: 20px;">
  <h2>Regression</h2>
</div>

Performs spatial panel regression (Common Effect, Fixed Effect, Random Effect) between land cover and socio-economic attributes. Outputs charts, coefficients, and interpretation text.

---

<div style="display: flex; align-items: center;">
  <img src="/icons/forecast.png" alt="Forecast" style="width: 100px; height: 100px; margin-right: 20px;">
  <h2>Estimation</h2>
</div>

Forecasts future land cover distribution using transition matrix and Markov-based projection, adaptable to user-defined time periods.

---

<div style="display: flex; align-items: center;">
  <img src="/icons/citytrend.png" alt="City Center Trend" style="width: 100px; height: 100px; margin-right: 20px;">
  <h2>City Center Trend Analysis</h2>
</div>

Tracks built-up center shifts across multiple land cover layers over time, displaying spatial movement patterns and directional flow.

---

<div style="display: flex; align-items: center;">
  <img src="/icons/localmorans.png" alt="Local Moran's I" style="width: 100px; height: 100px; margin-right: 20px;">
  <h2>Detecting Spatial Clusters and Outliers</h2>
</div>

Identifies statistically significant spatial clusters and outliers using Local Moran’s I. Detects HH (hotspot), LL (coldspot), HL, and LH patterns.

---

<div style="display: flex; align-items: center;">
  <img src="/icons/lq.png" alt="LQ Analysis" style="width: 100px; height: 100px; margin-right: 20px;">
  <h2>Economic Sector Specialization (LQ)</h2>
</div>

Calculates the Location Quotient (LQ) to evaluate sectoral dominance and specialization within a region compared to the overall average.

---

<div style="display: flex; align-items: center;">
  <img src="/icons/globalmorans.png" alt="Global Moran's I" style="width: 100px; height: 100px; margin-right: 20px;">
  <h2>Global Spatial Pattern Analysis</h2>
</div>

Measures global spatial autocorrelation using Moran’s I statistic. Identifies whether the distribution of values is clustered, dispersed, or random, with significance evaluation.

---

## 📦 Output Types

- PNG: Graphs and visual plots
- CSV: Tables and model results
- Shapefile: Result layers with classifications
- Descriptive text: Automatically generated interpretations

---

## 📌 Recommended Use

SELCA is best suited for spatial planners, researchers, data analysts, and policy makers needing spatial insight into how environmental and socio-economic dynamics evolve.

---

## 🧾 License

This plugin is licensed under the GNU General Public License v2 (or later). See the LICENSE file for details.
