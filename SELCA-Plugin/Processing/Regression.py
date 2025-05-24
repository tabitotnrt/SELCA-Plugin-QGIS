import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects
import statsmodels.api as sm
from matplotlib.animation import FuncAnimation
from qgis.PyQt.QtWidgets import QLabel, QComboBox, QFileDialog
import matplotlib.pyplot as plt
import os
from qgis.gui import QgsFieldComboBox

def run_regression(dialog):
    master_data = dialog.master_data.copy()
    dialog.logWidget.append("Starting regression analysis...")

    observer_field = dialog.findChild(QComboBox, "observedField")
    if observer_field is None or not observer_field.currentText():
        dialog.findChild(QLabel, "label_14").setText("<b>Error: Please select an observed field</b>")
        return
        
    observed_field = observer_field.currentText()
    if observed_field not in master_data.columns:
        dialog.findChild(QLabel, "label_14").setText(f"<b>Error: Column '{observed_field}' not found in master_data</b>")
        return

    try:
        master_data[observed_field] = master_data[observed_field].replace({',': ''}, regex=True).astype(float)
        dialog.logWidget.append(f"{observed_field} values: {master_data[observed_field].unique()}")
    except Exception as e:
        dialog.findChild(QLabel, "label_14").setText(f"<b>Error converting {observed_field}: {str(e)}</b>")
        return

    dependent_field = dialog.aliasDependent.text() if dialog.aliasDependent.text() else "Dependent"
    if dependent_field not in master_data.columns:
        dialog.findChild(QLabel, "label_14").setText(f"<b>Error: Column '{dependent_field}' not found in master_data</b>")
        return

    try:
        master_data[dependent_field] = pd.to_numeric(master_data[dependent_field], errors='coerce')
        dialog.logWidget.append(f"{dependent_field} values: {master_data[dependent_field].unique()}")
    except Exception as e:
        dialog.findChild(QLabel, "label_14").setText(f"<b>Error converting {dependent_field}: {str(e)}</b>")
        return

    time_stamps = master_data['Time Stamp'].unique()
    time_map = {stamp: idx + 1 for idx, stamp in enumerate(sorted(time_stamps))}
    master_data['Tahun'] = master_data['Time Stamp'].map(time_map)

    if master_data[observed_field].isna().all() or master_data[observed_field].nunique() <= 1:
        dialog.findChild(QLabel, "label_14").setText(f"<b>Error: {observed_field} has no valid values or no variation</b>")
        dialog.logWidget.append(f"{observed_field} statistics: {master_data[observed_field].describe()}")
        return

    kelas_columns = [col for col in master_data.columns if col.startswith("Kelas_") and col != observed_field]
    master_data["Non_Observed_Area"] = master_data[kelas_columns].sum(axis=1)

    if master_data[observed_field].dropna().nunique() <= 1:
        dialog.findChild(QLabel, "label_14").setText(f"<b>Error: {observed_field} has no variation or all NaN</b>")
        return

    correlations = []
    for city in master_data["Region"].unique():
        df_city = master_data[master_data["Region"] == city].copy()
        if len(df_city) > 1:
            valid_city = df_city.dropna(subset=[dependent_field, observed_field])
            if len(valid_city) > 1:
                corr_observed, p_observed = pearsonr(valid_city[dependent_field], valid_city[observed_field])
                correlations.append({
                    "Region": city,
                    f"{dependent_field} vs {observed_field} Correlation": corr_observed,
                    f"{dependent_field} vs {observed_field} P-value": p_observed
                })

    df_correlations = pd.DataFrame(correlations)
    avg_corr_observed = df_correlations[f"{dependent_field} vs {observed_field} Correlation"].mean() if not df_correlations.empty else np.nan
    dialog.findChild(QLabel, "labelPearson").setText(f"<b>{avg_corr_observed:.3f}</b>")

    # Classify the correlation strength
    if not np.isnan(avg_corr_observed):
        if abs(avg_corr_observed) >= 0.9:
            corr_classification = "Very High"
        elif abs(avg_corr_observed) >= 0.7:
            corr_classification = "High"
        elif abs(avg_corr_observed) >= 0.5:
            corr_classification = "Moderate"
        elif abs(avg_corr_observed) >= 0.3:
            corr_classification = "Low"
        else:
            corr_classification = "Very Low"
        descriptive_text = f"The result shows that the class {observed_field} have {corr_classification} correlation to the {dependent_field} "
    else:
        descriptive_text = "Average Correlation: N/A (No valid data)"
    
    description_label = dialog.findChild(QLabel, "descriptionLabel")
    if description_label:
        description_label.setText(f"<b>{descriptive_text}</b>")
    else:
        dialog.logWidget.append("Error: QLabel 'descriptionLabel' not found.")

    df_all = master_data.set_index(["Region", "Tahun"])
    y_all = df_all[dependent_field]
    X_all = df_all[[observed_field]]
    X_all = sm.add_constant(X_all)

    valid_data = pd.concat([y_all, X_all], axis=1).dropna()
    if len(valid_data) < 2:
        dialog.findChild(QLabel, "label_14").setText("<b>Error: Insufficient valid data for regression</b>")
        return

    y_all = valid_data[dependent_field]
    X_all = valid_data[["const", observed_field]]

    method = dialog.inputMethods.currentIndex()
    try:
        if method == 0:
            model = PooledOLS(y_all, X_all, check_rank=False)
            label = "CEM Fit"
        elif method == 1:
            model = PanelOLS(y_all, X_all, entity_effects=True, time_effects=False, check_rank=False)
            label = "FEM Fit"
        else:
            model = RandomEffects(y_all, X_all, check_rank=False)
            label = "REM Fit"

        results = model.fit()
        intercept = results.params.get("const", np.nan)
        coef_observed = results.params.get(observed_field, np.nan)
        equation = f"{dependent_field} = {intercept:.3f} + {coef_observed:.6f} * {observed_field}"
        dialog.findChild(QLabel, "label_14").setText(f"<b>{equation}</b>")
    except ValueError as e:
        dialog.findChild(QLabel, "label_14").setText(f"<b>Error: {str(e)}</b>")
        return

    # Store results for export
    dialog.regression_results = {
        "master_data": master_data,
        "observed_field": observed_field,
        "dependent_field": dependent_field,
        "intercept": intercept,
        "coef_observed": coef_observed,
        "avg_corr_observed": avg_corr_observed,
        "label": label
    }

    # Plotting on canvas (unchanged)
    fig = dialog.canvas.figure
    ax = fig.gca()
    ax.clear()
    scatter = ax.scatter([], [], color="#2E86AB", alpha=0.6, s=50, edgecolor='w', label="Data Points")
    line, = ax.plot([], [], color="#F46036", linestyle="-", linewidth=2, label=label)
    ax.set_xlabel(f"{observed_field} (km²)", fontsize=10, weight="medium")
    ax.set_ylabel(f"{dependent_field}", fontsize=10, weight="medium")
    ax.set_title(f"Impact of {observed_field} on {dependent_field}", fontsize=12, weight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9, frameon=False, labelspacing=0.5)
    ax.grid(True, linestyle="--", alpha=0.2, color="#D3D3D3")
    ax.set_facecolor("white")
    fig.set_facecolor("white")

    master_data_sorted = master_data.dropna(subset=[observed_field, dependent_field]).sort_values(by=observed_field)
    if master_data_sorted.empty:
        dialog.findChild(QLabel, "label_14").setText("<b>Error: No valid data for animation</b>")
        return

    def init():
        scatter.set_offsets(np.zeros((0, 2)))
        line.set_data([], [])
        ax.set_xlim(master_data_sorted[observed_field].min() * 0.95, master_data_sorted[observed_field].max() * 1.05)
        ax.set_ylim(master_data_sorted[dependent_field].min() * 0.95, master_data_sorted[dependent_field].max() * 1.05)
        return scatter, line

    def update(frame):
        if ax is None:
            print("Error: ax became None during animation")
            return scatter, line
        num_points = min(frame + 1, len(master_data_sorted))
        if num_points > 0:
            data = master_data_sorted.iloc[:num_points]
            scatter.set_offsets(np.c_[data[observed_field], data[dependent_field]])
        if num_points >= 2 and not np.isnan(intercept) and not np.isnan(coef_observed):
            x_current = np.linspace(master_data_sorted[observed_field].min(), data[observed_field].max(), 100)
            y_current = intercept + coef_observed * x_current
            line.set_data(x_current, y_current)
        return scatter, line

    total_plotting_time = 10
    interval = 20
    total_frames = int(total_plotting_time * 1000 / interval)

    ax.tick_params(axis="both", which="major", labelsize=9, color="#666666")
    dialog.canvas.draw()

    dialog.ani = FuncAnimation(fig, update, frames=total_frames,
                               init_func=init, blit=False, interval=interval, repeat=False)
    
    dialog.canvas.figure.canvas.draw_idle()
    dialog.canvas.figure.canvas.flush_events()

def export_regression_data(dialog):
    """Export regression plot with equation and Pearson correlation as a PNG, plus CSV data."""
    if not hasattr(dialog, 'regression_results'):
        dialog.logWidget.append("Error: Run regression analysis first.")
        dialog.progressBar.setValue(0)
        return

    dialog.logWidget.append("Exporting regression data...")
    dialog.progressBar.setValue(0)

    # Open folder selection dialog
    folder = QFileDialog.getExistingDirectory(dialog, "Select Folder to Save Regression Data")
    if not folder:
        dialog.logWidget.append("Export canceled: No folder selected.")
        dialog.progressBar.setValue(0)
        return

    try:
        # Extract regression results
        results = dialog.regression_results
        master_data = results["master_data"]
        observed_field = results["observed_field"]
        dependent_field = results["dependent_field"]
        intercept = results["intercept"]
        coef_observed = results["coef_observed"]
        avg_corr_observed = results["avg_corr_observed"]
        label = results["label"]

        # Create a new figure for export
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        master_data_sorted = master_data.dropna(subset=[observed_field, dependent_field]).sort_values(by=observed_field)
        
        # Plot scatter and regression line
        ax.scatter(master_data_sorted[observed_field], master_data_sorted[dependent_field], 
                   color="#2E86AB", alpha=0.6, s=50, edgecolor='w', label="Data Points")
        x_range = np.linspace(master_data_sorted[observed_field].min(), master_data_sorted[observed_field].max(), 100)
        y_range = intercept + coef_observed * x_range
        ax.plot(x_range, y_range, color="#F46036", linestyle="-", linewidth=2, label=label)
        
        # Customize plot
        ax.set_xlabel(f"{observed_field} (km²)", fontsize=10, weight="medium")
        ax.set_ylabel(f"{dependent_field}", fontsize=10, weight="medium")
        ax.set_title(f"Impact of {observed_field} on {dependent_field}", fontsize=12, weight="bold", pad=10)
        
        # Move legend to bottom-right to avoid overlap with text
        ax.legend(loc="lower right", fontsize=9, frameon=False, labelspacing=0.5)
        ax.grid(True, linestyle="--", alpha=0.2, color="#D3D3D3")
        ax.set_facecolor("white")
        fig.set_facecolor("white")
        
        # Add equation and Pearson correlation to the plot, positioned at the top-right
        equation_text = f"{dependent_field} = {intercept:.3f} + {coef_observed:.6f} * {observed_field}"
        corr_text = f"Pearson Correlation: {avg_corr_observed:.3f}"
        ax.text(0.95, 0.95, f"{equation_text}\n{corr_text}", transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8))

        # Adjust layout to prevent text overlap, adding padding at the top
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(top=0.85)

        # Save PNG
        png_path = os.path.join(folder, "Regression_Plot.png")
        fig.savefig(png_path, bbox_inches='tight', dpi=120)
        dialog.logWidget.append(f"Saved Regression Plot PNG to {png_path}")
        dialog.progressBar.setValue(50)

        # Export CSV
        csv_data = master_data[[observed_field, dependent_field, "Region", "Time Stamp"]].dropna()
        csv_path = os.path.join(folder, "Regression_Data.csv")
        csv_data.to_csv(csv_path, index=False)
        dialog.logWidget.append(f"Saved Regression Data CSV to {csv_path}")
        dialog.progressBar.setValue(100)

        dialog.logWidget.append("Export completed successfully.")
        plt.close(fig)  # Close the figure to free memory
    except Exception as e:
        dialog.logWidget.append(f"Error during export: {str(e)}")
        dialog.progressBar.setValue(0)