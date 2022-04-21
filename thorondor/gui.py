import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
import math
import pickle
import tables as tb
from prettytable import PrettyTable

import lmfit
from lmfit import minimize, Parameters, Parameter
from lmfit.models import *
import corner

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import display, Markdown, Latex, clear_output

from scipy import interpolate, optimize, sparse
from scipy.signal import savgol_filter
from scipy.stats import chisquare

from datetime import datetime

from thorondor.gui_iterable import DiamondDataset

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Legend, RangeTool, HoverTool, WheelZoomTool, CrosshairTool
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, export_png
from collections import defaultdict

output_notebook()


class Interface():
    """
    This  class is a Graphical User Interface (gui) that is meant to be used
    to process important amount of XPS datasets.

    This class makes extensive use of the ipywidgets and is thus meant to
    be used with a jupyter notebook.
    """

    def __init__(self, class_list=False):
        """
        All the widgets for the GUI are defined here.
        Two different initialization procedures are possible depending on
        whether or not a class_list is given in entry.
        """

        # Temporaty for energy values
        self.new_energy_column = np.round(np.linspace(-100, 1000, 2001), 2)
        self.interpol_step = 0.05

        # Filtering function parameters
        self.filter_window = 21
        self.filter_poly_order = 3

        # Plot parameters that do not change a lot
        self.legend = "conditions"
        self.figure_height = 400
        self.figure_width = 900
        self.matplotlib_colours = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22',
            '#17becf']

        # Initialise list
        self.class_list = []

        # Subdirectories
        directories = []
        for f in sorted([x[0] + "/" for x in os.walk(os.getcwd())]):
            if "/Classes/" in f or "/ExportData/" in f or "/Figures/" in f \
                    or ".ispyb/" in f or "/.ipynb_checkpoints/" in f:
                pass
            else:
                directories.append(f)

        # Widgets for the initialization
        self._list_widgets_init = interactive(
            self.class_list_init,
            data_folder=widgets.Dropdown(
                options=directories,
                value=os.getcwd() + "/",
                placeholder=os.getcwd() + "/",
                description='Data folder:',
                continuous_update=False,
                layout=Layout(width='90%'),
                style={'description_width': 'initial'}),
            tool=widgets.ToggleButtons(
                options=[
                    ("Neutral", "default"),
                    ("Start working", "work"),
                    ("Reload data", "reload"),
                    ("Delete all data", "delete"),
                ],
                value="default",
                description="Options",
                disabled=False,
                button_style="",
                style={'description_width': 'initial'}))
        self._list_widgets_init.children[1].observe(
            self.tool_handler, names="value")

        self.tab_init = widgets.VBox([
            self._list_widgets_init.children[0],
            self._list_widgets_init.children[1],
            self._list_widgets_init.children[-1]
        ])

        # Widgets for the data visualisation
        self.tab_data = interactive(
            self.print_data,
            used_dataset=widgets.Dropdown(
                options=self.class_list,
                description='Select the Dataset:',
                disabled=True,
                style={'description_width': 'initial'},
                layout=Layout(width='60%')),
            df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines"),
                    ("Fitted data", "fit_df")
                ],
                value="df",
                description='Select the dataframe:',
                disabled=True,
                style={'description_width': 'initial'}))

        # Widgets for the tools
        self.tab_tools = interactive(
            self.treat_data,
            method=widgets.ToggleButtons(
                options=[
                    ("Shift correction", "shift"),
                    ("Normalize", "norm"),
                    ("Deglitching", "deglitching"),
                    ("Determine errors", "errors"),
                    ("Linear Combination Fit", "LCF"),
                    # ("Save as .nxs (NeXuS)", "nexus"),
                ],
                value="shift",
                description='Tools:',
                disabled=True,
                button_style="",
                style={'description_width': 'initial'}),
            plot_bool=widgets.Checkbox(
                value=False,
                description='Fix tool',
                disabled=True,
                style={'description_width': 'initial'}))
        self.tab_tools.children[1].observe(
            self.tools_bool_handler, names="value")

        self._list_shift = interactive(
            self.energy_shift,
            used_dataset=widgets.Dropdown(
                options=self.class_list,
                description="Choose dataset :",
                disabled=False,
                style={
                    'description_width': 'initial'},
                layout=Layout(width='60%')),
            df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines"),
                    ("Fitted data", "fit_df")
                ],
                value="df",
                description='Use the dataframe:',
                disabled=False,
                style={'description_width': 'initial'}),
            x=widgets.Dropdown(
                options=[
                    ("Binding energy", "binding_energy"),
                    ("Kinetic energy", "kinetic_energy")
                ],
                value="binding_energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("RMS", "RMS"),
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_shift = widgets.VBox([
            self._list_shift.children[0],
            widgets.HBox(self._list_shift.children[1:4]),
            self._list_shift.children[-1]
        ])

        self._list_deglitching = interactive(
            self.correction_deglitching,
            used_dataset=widgets.Dropdown(
                options=self.class_list,
                description='Select the Dataset:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="60%")),
            df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines"),
                    ("Fitted data", "fit_df")
                ],
                value="df",
                description='Use the dataframe:',
                disabled=False,
                layout=Layout(width="50%"),
                style={'description_width': 'initial'}),
            pts=widgets.BoundedIntText(
                value=5,
                min=1,
                max=20,
                step=1,
                description="Nb of extra points",
                layout=Layout(width="50%"),
                style={'description_width': 'initial'},
                disabled=False),
            x=widgets.Dropdown(
                options=[
                    ("Binding energy", "binding_energy"),
                    ("Kinetic energy", "kinetic_energy")
                ],
                value="binding_energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("Weights", "weights"),
                    ("RMS", "RMS"),
                    ("User error", "user_error")
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            order=widgets.Dropdown(
                options=[
                    ("Linear", "linear"),
                    ("Quadratic", "quadratic"),
                    ("Cubic", "cubic")
                ],
                value="linear",
                description='Choose an order:',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_deglitching = widgets.VBox([
            self._list_deglitching.children[0],
            widgets.HBox(self._list_deglitching.children[1:3]),
            widgets.HBox(self._list_deglitching.children[3:6]),
            self._list_deglitching.children[-1]])

        self._list_errors_extraction = interactive(
            self.errors_extraction,
            used_dataset=widgets.Dropdown(
                options=self.class_list,
                description='Dataset:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width='60%')),
            df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines"),
                    ("Fitted data", "fit_df")
                ],
                value="df",
                description='Dataframe:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="60%")),
            x_axis=widgets.Dropdown(
                options=[
                    ("Binding energy", "binding_energy"),
                    ("Kinetic energy", "kinetic_energy")
                ],
                value="binding_energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y_axis=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("Weights", "weights"),
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            nbpts=widgets.Dropdown(
                options=[i for i in range(5, 20)],
                value=13,
                description='Nb of points per interval:',
                disabled=False,
                style={'description_width': 'initial'}),
            deg=widgets.IntSlider(
                value=2,
                min=0,
                max=3,
                step=1,
                description='Degree:',
                disabled=False,
                orientation="horizontal",
                continuous_update=False,
                readout=True,
                readout_format="d",
                style={'description_width': 'initial'}),
            direction=widgets.Dropdown(
                options=[("Left", "left"), ("Right", "right")],
                value="left",
                description='Direction if odd:',
                disabled=False,
                style={'description_width': 'initial'}),
            compute=widgets.Checkbox(
                value=False,
                description='Compute errors',
                disabled=False,
                style={'description_width': 'initial'}))
        self._list_errors_extraction.children[7].observe(
            self.error_extraction_handler, names="value")
        self.widget_list_errors_extraction = widgets.VBox([
            self._list_errors_extraction.children[0],
            widgets.HBox(self._list_errors_extraction.children[1:3]),
            widgets.HBox(self._list_errors_extraction.children[3:7]),
            self._list_errors_extraction.children[-2],
            self._list_errors_extraction.children[-1]
        ])

        self._list_LCF = interactive(
            self.LCF,
            ref_spectra=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[0:2],
                description='Select at least two references for LCF:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column', width="50%")),
            spec_number=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[2:5],
                rows=5,
                description='Spectra to analyze:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column', width="50%")),
            used_dataset=widgets.Dropdown(
                options=self.class_list,
                description='Dataset to plot between the Datasets to analyze:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width='60%')),
            df_type=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines"),
                    ("Fitted data", "fit_df")
                ],
                value="reduced_df",
                description='Pick the dataframe:',
                disabled=False,
                style={'description_width': 'initial'}),
            x=widgets.Dropdown(
                options=[
                    ("Binding energy", "binding_energy"),
                    ("Kinetic energy", "kinetic_energy")
                ],
                value="binding_energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                ],
                value="second_normalized_\u03BC",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            LCF_bool=widgets.Checkbox(
                value=False,
                description='Perform LCF',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_LCF = widgets.VBox([
            widgets.HBox(self._list_LCF.children[:2]),
            widgets.HBox(self._list_LCF.children[2:4]),
            widgets.HBox(self._list_LCF.children[4:6]),
            self._list_LCF.children[-2],
            self._list_LCF.children[-1]
        ])

        self._list_norm_data = interactive(
            self.norm_data,
            used_dataset=widgets.Dropdown(
                options=self.class_list,
                description='Choose dataset:',
                disabled=False,
                style={
                    'description_width': 'initial'},
                layout=Layout(width='60%')),
            df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines"),
                    ("Fitted data", "fit_df")
                ],
                value="df",
                description='Use the dataframe:',
                disabled=False,
                style={'description_width': 'initial'}),
            x=widgets.Dropdown(
                options=[
                    ("Binding energy", "binding_energy"),
                    ("Kinetic energy", "kinetic_energy")
                ],
                value="binding_energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("RMS", "RMS"),
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_norm_data = widgets.VBox([
            self._list_norm_data.children[0],
            widgets.HBox(self._list_norm_data.children[1:4]),
            self._list_norm_data.children[-1]
        ])

        # Widgets for the Reduction
        self._list_tab_reduce_method = interactive(
            self.reduce_data,
            method=widgets.ToggleButtons(
                options=[
                    ("Least square method", "LSF"),
                    ("Chebyshev polynomials", "Chebyshev"),
                    ("Polynoms", "Polynoms"),
                    ("Single Spline", "SingleSpline"),
                    ("Splines", "Splines"),
                    #("Normalize by maximum", "NormMax")
                ],
                value="LSF",
                description='Pick reduction method:',
                disabled=True,
                button_style="",
                tooltips=[
                    'Least Square method',
                    'Chebyshev Polynomials',
                    "Multiple polynoms derived on short Intervals",
                    "Subtraction of a spline",
                    "Pre-edge and post-edge splines determination",
                    "Normalize spectra by maximum intensity value"
                ],
                style={'description_width': 'initial'}),
            used_class_list=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[0:1],
                rows=5,
                description='Select all the datasets to reduce together:',
                disabled=True,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column')),
            used_dataset=widgets.Dropdown(
                options=self.class_list,
                description="Dataset to plot between the Datasets to analyze:",
                disabled=True,
                style={'description_width': 'initial'},
                layout=Layout(width="60%")),
            df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines"),
                    ("Fitted data", "fit_df")
                ],
                value="shifted_df",
                description='Select the dataframe:',
                disabled=True,
                style={'description_width': 'initial'}),
            plot_bool=widgets.Checkbox(
                value=False,
                description='Start reduction',
                disabled=True,
                style={'description_width': 'initial'}))
        self._list_tab_reduce_method.children[4].observe(
            self.reduce_bool_handler, names="value")

        self.tab_reduce_method = widgets.VBox([
            self._list_tab_reduce_method.children[0],
            self._list_tab_reduce_method.children[1],
            self._list_tab_reduce_method.children[2],
            widgets.HBox(self._list_tab_reduce_method.children[3:5]),
            self._list_tab_reduce_method.children[-1]
        ])

        # Widgets for the LSF background reduction and normalization method
        self._list_reduce_LSF = interactive(
            self.reduce_LSF,
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            interval=widgets.FloatRangeSlider(
                min=self.new_energy_column[0],
                value=[
                    self.new_energy_column[0], self.new_energy_column[-1]],
                max=self.new_energy_column[-1],
                step=self.interpol_step,
                description='Energy range (eV):',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format='.2f',
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            lam=widgets.IntSlider(
                value=10**7,
                min=10**4,
                max=10**7,
                step=1,
                description='lambda:',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
                style={'description_width': 'initial'}),
            p=widgets.IntSlider(
                value=2,
                min=1,
                max=100,
                step=1,
                description='p:',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
                style={'description_width': 'initial'}))
        self.widget_list_reduce_LSF = widgets.VBox([
            self._list_reduce_LSF.children[0],
            self._list_reduce_LSF.children[1],
            widgets.HBox(self._list_reduce_LSF.children[2:4]),
            self._list_reduce_LSF.children[-1]
        ])

        # Widgets for the chebyshev background reduction and normalization method
        self._list_reduce_chebyshev = interactive(
            self.reduce_chebyshev,
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            interval=widgets.FloatRangeSlider(
                min=self.new_energy_column[0],
                value=[
                    self.new_energy_column[0], self.new_energy_column[-1]],
                max=self.new_energy_column[-1],
                step=self.interpol_step,
                description='Energy range (eV):',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format='.2f',
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            p=widgets.IntSlider(
                value=10,
                min=0,
                max=100,
                step=1,
                description='Degree of Polynomials:',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
                style={'description_width': 'initial'}),
            n=widgets.IntSlider(
                value=2,
                min=1,
                max=10,
                step=1,
                description='Importance of weights ',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
                style={'description_width': 'initial'}))
        self.widget_list_reduce_chebyshev = widgets.VBox([
            self._list_reduce_chebyshev.children[0],
            self._list_reduce_chebyshev.children[1],
            widgets.HBox(self._list_reduce_chebyshev.children[2:4]),
            self._list_reduce_chebyshev.children[-1]
        ])

        # Widgets for the Polynoms background reduction and normalization method
        self._list_reduce_polynoms = interactive(
            self.reduce_polynoms,
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            interval=widgets.FloatRangeSlider(
                min=self.new_energy_column[0],
                value=[
                    self.new_energy_column[0], self.new_energy_column[-1]],
                max=self.new_energy_column[-1],
                step=self.interpol_step,
                description='Energy range (eV):',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format='.2f',
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            sL=widgets.BoundedIntText(
                value=4,
                min=4,
                max=11,
                step=1,
                description='Slider pts::',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_reduce_polynoms = widgets.VBox([
            self._list_reduce_polynoms.children[0],
            self._list_reduce_polynoms.children[1],
            self._list_reduce_polynoms.children[2],
            self._list_reduce_polynoms.children[-1]
        ])

        # Widgets for the single spline background reduction and normalization method
        self._list_reduce_single_spline = interactive(
            self.reduce_single_spline,
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            order=widgets.Dropdown(
                options=[
                    ("Select and order", "value"),
                    ("Victoreen", "victoreen"),
                    ("0", 0),
                    ("1", 1),
                    ("2", 2),
                    ("3", 3)
                ],
                value="value",
                description='Order:',
                disabled=False,
                style={'description_width': 'initial'}),
            interval=widgets.FloatRangeSlider(
                min=self.new_energy_column[0],
                value=[self.new_energy_column[0], np.round(
                    self.new_energy_column[0] + 0.33*(self.new_energy_column[-1] - self.new_energy_column[0]), 0)],
                max=self.new_energy_column[-1],
                step=self.interpol_step,
                description='Energy range (eV):',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format='.2f',
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            cursor=widgets.FloatSlider(
                value=np.round(self.new_energy_column[0] + 0.45*(
                    self.new_energy_column[-1] - self.new_energy_column[0]), 0),
                step=self.interpol_step,
                min=self.new_energy_column[0],
                max=self.new_energy_column[-1],
                description='Cursor:',
                orientation="horizontal",
                continuous_update=False,
                readout=True,
                readout_format='.2f',
                style={'description_width': 'initial'},
                disabled=False),
            param_A=widgets.Text(
                value="1000000000",
                placeholder='A = ',
                description='A:',
                disabled=True,
                continuous_update=False,
                style={'description_width': 'initial'}),
            param_B=widgets.Text(
                value="1000000000",
                placeholder='B = ',
                description='B:',
                disabled=True,
                continuous_update=False,
                style={'description_width': 'initial'}))
        self.widget_list_reduce_single_spline = widgets.VBox([
            widgets.HBox(self._list_reduce_single_spline.children[:2]),
            self._list_reduce_single_spline.children[2],
            self._list_reduce_single_spline.children[3],
            widgets.HBox(self._list_reduce_single_spline.children[4:6]),
            self._list_reduce_single_spline.children[-1]
        ])
        self._list_reduce_single_spline.children[1].observe(
            self.param_victoreen_handler_single, names="value")

        # Widgets for the Splines background reduction and normalization method
        self._list_reduce_splines_derivative = interactive(
            self.reduce_splines_derivative,
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            interval=widgets.FloatRangeSlider(
                min=self.new_energy_column[0],
                value=[
                    self.new_energy_column[0], self.new_energy_column[-1]],
                max=self.new_energy_column[-1],
                step=self.interpol_step,
                description='Energy range (eV):',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format='.2f',
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')))
        self.widget_list_reduce_splines_derivative = widgets.VBox(
            [self._list_reduce_splines_derivative.children[0], self._list_reduce_splines_derivative.children[1], self._list_reduce_splines_derivative.children[-1]])

        # Widgets for the LSF background reduction and normalization method
        self._list_normalize_maxima = interactive(
            self.normalize_maxima,
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            interval=widgets.FloatRangeSlider(
                min=self.new_energy_column[0],
                value=[
                    self.new_energy_column[0], self.new_energy_column[-1]],
                max=self.new_energy_column[-1],
                step=self.interpol_step,
                description='Energy range (eV):',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format='.2f',
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')))
        self.widget_list_normalize_maxima = widgets.VBox([
            self._list_normalize_maxima.children[0],
            self._list_normalize_maxima.children[1],
            self._list_normalize_maxima.children[-1]
        ])

        # Widgets for the fit,
        self._list_define_fitting_df = interactive(
            self.define_fitting_df,
            used_dataset=widgets.Dropdown(
                options=self.class_list,
                description='Select the Dataset:',
                disabled=True,
                style={'description_width': 'initial'},
                layout=Layout(width='60%')),
            df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines"),
                    ("Fitted data", "fit_df")
                ],
                value="df",
                description='Select the dataframe:',
                disabled=True,
                style={'description_width': 'initial'}),
            show_bool=widgets.Checkbox(
                value=False,
                description='Fix dataframe.',
                disabled=True,
                style={'description_width': 'initial'}))
        self._list_define_fitting_df.children[2].observe(
            self.fit_handler, names="value")
        self.tab_fit = widgets.VBox([
            widgets.HBox(self._list_define_fitting_df.children[:3]),
            self._list_define_fitting_df.children[-1]
        ])

        self._list_define_model = interactive(
            self.define_model,
            x_axis=widgets.Dropdown(
                options=[
                    ("Binding energy", "binding_energy"),
                    ("Kinetic energy", "kinetic_energy")
                ],
                value="binding_energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y_axis=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            interval=widgets.FloatRangeSlider(
                min=self.new_energy_column[0],
                value=[
                    self.new_energy_column[0], self.new_energy_column[-1]],
                max=self.new_energy_column[-1],
                step=self.interpol_step,
                description='Energy range (eV):',
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format='.2f',
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            peak_number=widgets.BoundedIntText(
                value=2,
                min=0,
                max=10,
                step=1,
                description='Amount of Peaks:',
                disabled=False,
                style={'description_width': 'initial'}),
            peak_type=widgets.Dropdown(
                options=[
                    ("Gaussian", GaussianModel),
                    ("Lorentzian", LorentzianModel),
                    ("Split Lorentzian", SplitLorentzianModel),
                    ("Voigt", VoigtModel),
                    ("Pseudo-Voigt", PseudoVoigtModel),
                    ("Moffat", MoffatModel),
                    ("Pearson7", Pearson7Model),
                    ("StudentsT", StudentsTModel),
                    ("Breit-Wigner", BreitWignerModel),
                    ("Log-normal", LognormalModel),
                    ("Exponential-Gaussian", ExponentialGaussianModel),
                    ("Skewed-Gaussian", SkewedGaussianModel),
                    ("Skewed-Voigt", SkewedVoigtModel),
                    ("Donaich", DoniachModel)
                ],
                value=LorentzianModel,
                description='Peak distribution:',
                disabled=False,
                style={'description_width': 'initial'}),
            background_type=widgets.Dropdown(
                options=[
                    ("Constant", ConstantModel),
                    ("Linear", LinearModel),
                    ("Victoreen", "victoreen"),
                    ("Quadratic", QuadraticModel),
                    ("Polynomial", PolynomialModel)
                ],
                value=ConstantModel,
                description='Background mode:',
                disabled=False,
                style={'description_width': 'initial'}),
            pol_degree=widgets.IntSlider(
                value=3,
                min=0,
                max=7,
                step=1,
                description='Degree:',
                disabled=True,
                orientation="horizontal",
                continuous_update=False,
                readout=True,
                readout_format="d",
                style={'description_width': 'initial'}),
            step_type=widgets.Dropdown(
                options=[
                    ("No step", False),
                    ("linear", "linear"),
                    ("arctan", "arctan"),
                    ("erf", "erf"),
                    ("logistic", "logistic")
                ],
                value=False,
                description='Step mode:',
                disabled=False,
                style={'description_width': 'initial'}),
            method=widgets.Dropdown(
                options=[
                    ("Levenberg-Marquardt", "leastsq"),
                    ("Nelder-Mead", "nelder"),
                    ("L-BFGS-B", "lbfgsb"),
                    ("BFGS", "bfgs"),
                    ("Maximum likelihood via Monte-Carlo Markov Chain", "emcee")
                ],
                value="leastsq",
                description='Pick a minimization method (read doc first):',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="50%")),
            w=widgets.Dropdown(
                options=[
                    ("No (=1)", False),
                    ("\u03BC Variance (1/\u03BC)", "\u03BC_variance"),
                    ("RMS", "RMS"),
                    ("User error", "user_error")
                ],
                value=False,
                description='Use weights.',
                disabled=False,
                style={'description_width': 'initial'}),
            fix_model=widgets.Checkbox(
                value=False,
                description='Fix Model.',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_define_model = widgets.VBox([
            widgets.HBox(self._list_define_model.children[0:2]), self._list_define_model.children[2], widgets.HBox(
                self._list_define_model.children[3:5]),
            widgets.HBox(self._list_define_model.children[5:8]), widgets.HBox(
                self._list_define_model.children[8:11]), self._list_define_model.children[-1]
        ])
        self._list_define_model.children[10].observe(
            self.model_handler, names="value")
        self._list_define_model.children[5].observe(
            self.model_degree_handler, names="value")

        # Widgets for the plotting
        self._list_plot_dataset = interactive(
            self.plot_dataset,
            spec_number=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[0:1],
                rows=5,
                description='Spectra to plot:',
                disabled=True,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column')),
            plot_df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines"),
                    ("Fitted data", "fit_df")
                ],
                value="df",
                description='Use the dataframe:',
                disabled=True,
                style={'description_width': 'initial'}),
            x=widgets.Dropdown(
                options=[
                    ("Binding energy", "binding_energy"),
                    ("Kinetic energy", "kinetic_energy")
                ],
                value="binding_energy",
                description='Pick an x-axis',
                disabled=True,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Intensity", "intensity"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("Weights", "weights"),
                    ("RMS", "RMS"),
                    ("User error", "user_error")
                ],
                value="intensity",
                description='Pick an y-axis',
                disabled=True,
                style={'description_width': 'initial'}),
            x_axis=widgets.Text(
                value="Binding energy",
                placeholder="Binding energy",
                description='Type the name of the x axis:',
                disabled=True,
                continuous_update=False,
                style={'description_width': 'initial'}),
            y_axis=widgets.Text(
                value='Intensity',
                placeholder='Intensity',
                description='Type the name of the y axis:',
                disabled=True,
                continuous_update=False,
                style={'description_width': 'initial'}),
            title=widgets.Text(
                value='Plot',
                placeholder='Plot',
                description='Type the title you wish to use:',
                disabled=True,
                continuous_update=False,
                style={'description_width': 'initial'}),
            check_plot=widgets.ToggleButtons(
                options=[('Clear', "Zero"), ('Plot', "Plot")],
                value="Zero",
                description='Plot:',
                disabled=True,
                button_style="",
                tooltips=[
                    'Nothing is plotted',
                    'We plot one Dataset',
                    'We plot all the spectra'
                ],
                style={'description_width': 'initial'}))
        self.tab_plot = widgets.VBox([
            self._list_plot_dataset.children[0],
            widgets.HBox(self._list_plot_dataset.children[1:4]),
            widgets.HBox(self._list_plot_dataset.children[4:7]),
            self._list_plot_dataset.children[-2],
            self._list_plot_dataset.children[-1]
        ])

        # Create the final window
        self.window = widgets.Tab(children=[
            self.tab_init,
            self.tab_data,
            self.tab_tools,
            self.tab_reduce_method,
            self.tab_fit,
            self.tab_plot,
        ])
        self.window.set_title(0, 'Initialize')
        self.window.set_title(1, 'View Data')
        self.window.set_title(2, 'Tools')
        self.window.set_title(3, 'Reduce Background')
        self.window.set_title(4, 'Fit')
        self.window.set_title(5, 'Plot')

        # Display window
        if class_list:
            self._list_widgets_init.children[0].value = self.data_folder
            self._list_widgets_init.children[0].disabled = True
            self._list_widgets_init.children[1].value = "default"

            for w in self.tab_data.children[:-1]:
                w.disabled = False

            for w in self.tab_tools.children[:-1]:
                w.disabled = False

            for w in self._list_tab_reduce_method.children[:-1]:
                w.disabled = False

            for w in self._list_define_fitting_df.children[:-1]:
                w.disabled = False

            for w in self._list_plot_dataset.children[:-1]:
                w.disabled = False

            # Show the plotting first
            self.window.selected_index = 5

            display(self.window)

        elif not class_list:
            display(self.window)

    # Initialization interactive function, if no previous work had been done
    def class_list_init(
        self,
        data_folder,
        tool,
    ):
        """
        Function that generates or updates three subfolders in the "work_dir":
            _ data_folder where you will save your data files.
            _ data_folder/export_data where the raw data files will be saved
              (stripped of all metadata), in .txt format
            _ data_folder/classes where the data will be saved as a Dataset class
              at the end of your work.

        :param data_folder: root folder to store data in
        :param tool: Delete all processed data or start working
        """

        self.data_folder = data_folder
        path_classes = self.data_folder + "Classes/"
        path_data_as_csv = self.data_folder + "ExportData/"
        path_figures = self.data_folder + "Figures/"

        self.folders = [self.data_folder, path_classes,
                        path_data_as_csv, path_figures]

        if tool == "default":
            # Disable other widgets for now
            for w in self.tab_data.children[:-1] + \
                    self.tab_tools.children[:-1] + \
                    self._list_tab_reduce_method.children[:-1] + \
                    self._list_define_fitting_df.children[:-1] + \
                    self._list_plot_dataset.children[:-1]:
                w.disabled = True

            clear_output(True)

        elif tool == "delete":
            # Remove widgets options
            for w in self.tab_data.children[:-1] + \
                    self.tab_tools.children[:-1] + \
                    self._list_tab_reduce_method.children[:-1] + \
                    self._list_define_fitting_df.children[:-1] + \
                    self._list_plot_dataset.children[:-1]:
                w.disabled = True

            self.class_list, self.df_names = [], []

            self.tab_data.children[0].options = self.class_list
            self._list_shift.children[0].options = self.class_list
            self._list_norm_data.children[0].options = self.class_list
            self._list_deglitching.children[0].options = self.class_list
            self._list_errors_extraction.children[0].options = self.class_list
            self._list_LCF.children[0].options = self.class_list
            self._list_LCF.children[1].options = self.class_list
            self._list_LCF.children[2].options = self.class_list

            self._list_tab_reduce_method.children[1].options = self.class_list
            self._list_tab_reduce_method.children[2].options = self.class_list

            self._list_define_fitting_df.children[0].options = self.class_list

            self._list_plot_dataset.children[0].options = self.class_list

            self.tab_data.children[1].options = self.df_names
            self._list_shift.children[1].options = self.df_names
            self._list_norm_data.children[1].options = self.df_names
            self._list_deglitching.children[1].options = self.df_names
            self._list_errors_extraction.children[1].options = self.df_names
            self._list_LCF.children[3].options = self.df_names
            self._list_tab_reduce_method.children[3].options = self.df_names
            self._list_define_fitting_df.children[1].options = self.df_names
            self._list_plot_dataset.children[1].options = self.df_names

            # Delete data
            CleanedFolder = [path_classes, path_data_as_csv, path_figures]
            for f in CleanedFolder:
                for the_file in os.listdir(f):
                    file_path = os.path.join(f, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)

            print("Work has been reset")
            clear_output(True)

        elif tool == "work":
            for folder in self.folders:
                if not os.path.exists(folder):
                    try:
                        os.makedirs(folder)
                        print(f"{folder} well created.\n")
                    except FileExistsError:
                        print(f"{folder} already exists.\n")
                    except Exception as e:
                        raise e

            # Get filenames
            file_locations = [p.replace("\\", "/") for p in sorted(glob.glob(
                f"{self.folders[0]}*.nxs")
            )]

            # Initialize list of all the dataframes
            self.df_names, self.class_list = [], []

            # Add classes to classlist
            for f in file_locations:
                try:
                    # Create dataset
                    C = DiamondDataset(f)
                    self.class_list.append(C)
                    self.df_names = self.df_names + C.df_names

                    # Pickle dataset
                    C.pickle_dataset(
                        self.folders[1] +
                        C.filename.split(".")[0]+".pickle"
                    )
                except Exception as e:
                    raise e

            # Get the list of all the unique dataframes
            self.df_names = sorted(list(set(self.df_names)))

            # Update widgets status and options
            for w in self.tab_data.children[:-1] + \
                    self.tab_tools.children[:-1] + \
                    self._list_tab_reduce_method.children[:-1] + \
                    self._list_define_fitting_df.children[:-1] + \
                    self._list_plot_dataset.children[:-1]:
                w.disabled = False

            self.tab_data.children[0].options = self.class_list
            self._list_shift.children[0].options = self.class_list
            self._list_norm_data.children[0].options = self.class_list
            self._list_deglitching.children[0].options = self.class_list
            self._list_errors_extraction.children[0].options = self.class_list
            self._list_LCF.children[0].options = self.class_list
            self._list_LCF.children[1].options = self.class_list
            self._list_LCF.children[2].options = self.class_list

            self._list_tab_reduce_method.children[1].options = self.class_list
            self._list_tab_reduce_method.children[2].options = self.class_list

            self._list_define_fitting_df.children[0].options = self.class_list

            self._list_plot_dataset.children[0].options = self.class_list

            self.tab_data.children[1].options = self.df_names
            self._list_shift.children[1].options = self.df_names
            self._list_norm_data.children[1].options = self.df_names
            self._list_deglitching.children[1].options = self.df_names
            self._list_errors_extraction.children[1].options = self.df_names
            self._list_LCF.children[3].options = self.df_names
            self._list_tab_reduce_method.children[3].options = self.df_names
            self._list_define_fitting_df.children[1].options = self.df_names
            self._list_plot_dataset.children[1].options = self.df_names

        elif tool == "reload":
            # Get filenames
            file_locations = sorted(glob.glob(self.folders[1]+"*.pickle"))

            # Initialize list of all the dataframes
            self.df_names, self.class_list = [], []

            for f in file_locations:
                try:
                    C = DiamondDataset.unpickle(f)
                    self.class_list.append(C)
                    self.df_names = self.df_names + C.df_names
                    print("Loaded", f)
                    print(f"\t{C.df_names}")

                except EOFError:
                    print(
                        f"{C.filename} is empty."
                        "Restart the procedure from the beginning,"
                        "this may be due to a crash of Jupyter.")
                except FileNotFoundError:
                    print(f"The Class does not exist for {n}")

            # Get the list of all the unique dataframes
            self.df_names = sorted(list(set(self.df_names)))

            # Update widgets status and options
            for w in self.tab_data.children[:-1] + \
                    self.tab_tools.children[:-1] + \
                    self._list_tab_reduce_method.children[:-1] + \
                    self._list_define_fitting_df.children[:-1] + \
                    self._list_plot_dataset.children[:-1]:
                w.disabled = False

            self.tab_data.children[0].options = self.class_list
            self._list_shift.children[0].options = self.class_list
            self._list_norm_data.children[0].options = self.class_list
            self._list_deglitching.children[0].options = self.class_list
            self._list_errors_extraction.children[0].options = self.class_list
            self._list_LCF.children[0].options = self.class_list
            self._list_LCF.children[1].options = self.class_list
            self._list_LCF.children[2].options = self.class_list

            self._list_tab_reduce_method.children[1].options = self.class_list
            self._list_tab_reduce_method.children[2].options = self.class_list

            self._list_define_fitting_df.children[0].options = self.class_list

            self._list_plot_dataset.children[0].options = self.class_list

            self.tab_data.children[1].options = self.df_names
            self._list_shift.children[1].options = self.df_names
            self._list_norm_data.children[1].options = self.df_names
            self._list_deglitching.children[1].options = self.df_names
            self._list_errors_extraction.children[1].options = self.df_names
            self._list_LCF.children[3].options = self.df_names
            self._list_tab_reduce_method.children[3].options = self.df_names
            self._list_define_fitting_df.children[1].options = self.df_names
            self._list_plot_dataset.children[1].options = self.df_names

    # Visualization interactive function

    def print_data(self, used_dataset, df):
        """
        Print the main attributes of each DataFrame in the object and displays
        the DataFrame selected.

        :param used_dataset: Dataset used
        :param df: DataFrame used
        """

        # Print a pretty table of dataset attributes
        table = [
            ["DataFrame",
             "Count time (s)",
             "Iterations",
             "Data shape",
             "Step (eV)",
             "Pass E. (eV)",
             "Photon E. (eV)",
             # "Work function"
             "Norm range"]
        ]
        tab = PrettyTable(table[0])
        for s in used_dataset.df_names:
            try:
                tab.add_rows([[
                    s,
                    getattr(used_dataset, s[:-3]+"_count_time"),
                    int(getattr(used_dataset, s[:-3]+"_iterations")),
                    getattr(used_dataset, s[:-3]+"_spectra").shape,
                    getattr(used_dataset, s[:-3]+"_step_energy"),
                    int(getattr(used_dataset, s[:-3]+"_pass_energy")),
                    int(np.round(getattr(used_dataset,
                        s[:-3]+"_photon_energy"), 0)),
                    # getattr(used_dataset, s[:-3]+"_work_function"),
                    getattr(used_dataset, s[:-3]+"_norm_range"),
                ]])
            except AttributeError:
                pass
        print(tab)

        # Print shift if existing
        try:
            print(
                "\n##################################################################"
                f"\nEnergy shift common to all datasets: {used_dataset.shift:.2f} eV"
                "\n##################################################################"
            )
        except AttributeError:
            pass

        # Display df
        try:
            # display currently selected data frame
            display(getattr(used_dataset, df))

        except AttributeError:
            print(f"Wrong Dataset and column combination !")

    # Tools global interactive function

    def treat_data(self, method, plot_bool):
        """Lauch data reduction method depending on method"""
        if method == "shift" and plot_bool:
            display(self.widget_list_shift)
        if method == "deglitching" and plot_bool:
            display(self.widget_list_deglitching)
        if method == "errors" and plot_bool:
            display(self.widget_list_errors_extraction)
        if method == "LCF" and plot_bool:
            display(self.widget_list_LCF)
        if method == "norm" and plot_bool:
            display(self.widget_list_norm_data)
        if not plot_bool:
            print("Window cleared")
            clear_output(True)
            plt.close()

    def energy_shift(self, used_dataset, df, x, y):
        """
        Allows one to shift each Dataset by a certain amount to align on the
        Fermi energy, shift is performed on all the Dataset dataframes

        :param used_dataset: Dataset used
        :param df: DataFrame used
        :param x: x axis
        :param y: y axis
        """

        try:
            self.used_dataset = used_dataset
            self.used_df = getattr(used_dataset, df)  # not a copy !
            v1, v2 = min(self.used_df[x].values), max(self.used_df[x].values)

            @interact(
                interval=widgets.FloatRangeSlider(
                    value=[v1, v2],
                    min=v1,
                    max=v2,
                    step=0.5,
                    description='Range:',
                    disabled=False,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format="d",
                    style={'description_width': 'initial'},
                    layout=Layout(width="50%", height='40px')))
            def zoom_on_data(interval):
                # Get interval
                shift_df = self.used_df[
                    (self.used_df[x] > interval[0]) & (
                        self.used_df[x] < interval[1])
                ]

                # Smooth data
                y_smooth = savgol_filter(
                    shift_df[y],
                    self.filter_window,
                    self.filter_poly_order,
                )

                # Compute derivative
                yp = np.diff(y_smooth) / np.diff(shift_df[x])
                xp = (np.array(shift_df[x])[:-1] +
                      np.array(shift_df[x])[1:]) / 2

                # Get first guess of shift value
                shift = xp[np.where(yp == max(yp))]
                print(f"Initial guess for shift = {shift} eV")

                # Create sources
                source = ColumnDataSource(
                    data=dict(
                        x=shift_df[x],
                        y=shift_df[y],
                    ))

                source_smooth = ColumnDataSource(
                    data=dict(
                        x_smooth=shift_df[x],
                        y_smooth=y_smooth,
                    ))

                sourcep = ColumnDataSource(
                    data=dict(
                        xp=xp,
                        yp=yp,
                    ))

                # Create figure
                TOOLTIPS = [
                    (f"{x} (eV), {y}", "($x, $y)"),
                    ("index", "$index"),
                ]

                p = figure(
                    height=self.figure_height, width=self.figure_width,
                    tools="box_zoom, pan, wheel_zoom, reset, undo, redo, crosshair, hover, save",
                    tooltips=TOOLTIPS,
                    x_axis_label=x + " (eV)",
                    y_axis_label=y,
                    active_scroll="wheel_zoom",
                    active_drag="pan",
                )

                # Add line
                p.line("x", "y", source=source, legend_label="Data",
                       color=self.matplotlib_colours[0], line_alpha=0.8,
                       hover_line_alpha=1.0, hover_line_width=2.0,
                       muted_alpha=0.1)

                p.line("x_smooth", "y_smooth", source=source_smooth,
                       legend_label="Smoothed data",
                       color=self.matplotlib_colours[1], line_alpha=0.6,
                       hover_line_alpha=1.0, hover_line_width=2.0,
                       muted_alpha=0.1)

                # Add derivative plot
                p.line("xp", "yp", source=sourcep, legend_label='Derivative',
                       color=self.matplotlib_colours[2], line_alpha=0.8,
                       hover_line_alpha=1.0, hover_line_width=2.0,
                       muted_alpha=0.1)

                # Hide plot by clicking on legend
                p.legend.click_policy = "hide"

                # Show figure
                show(p)

                @interact(
                    shift_widget=widgets.FloatText(
                        value=shift,
                        description='Shift:',
                        disabled=False,
                        step=0.01,
                    ),
                    fix_shift_button=widgets.Checkbox(
                        value=False,
                        description="Fix shift.",
                        indent=False,
                        icon="check"))
                def FixShift(shift_widget, fix_shift_button):
                    "Fixes the shift"
                    if fix_shift_button:
                        clear_output(True)

                        # Save shift
                        try:
                            self.used_dataset.shift += shift_widget
                        except AttributeError:
                            self.used_dataset.shift = shift_widget

                        # Apply on all the dataframes
                        for df_name in self.used_dataset.df_names:
                            # Do not take a copy so that the shift is saved
                            df = getattr(self.used_dataset, df_name)
                            df.kinetic_energy -= shift_widget
                            df.binding_energy -= shift_widget

                            # Save as csv
                            self.save_df_to_csv(self.used_dataset, df_name)
                            print(
                                f"Shift ({shift_widget} eV) applied to {df_name} and saved as csv.")

                        # Pickle dataset
                        self.used_dataset.pickle_dataset(
                            self.folders[1] +
                            self.used_dataset.filename.split(".")[0]+".pickle"
                        )

        except (AttributeError, KeyError):
            print(f"Wrong Dataset and column combination !")

    def norm_data(self, used_dataset, df, x, y):
        """Allows one to normalize the Dataset"""

        try:
            self.used_dataset = used_dataset
            self.used_df_name = df
            self.used_df = getattr(used_dataset, df)  # not a copy !
            v1, v2 = min(self.used_df[x].values), max(self.used_df[x].values)

            @interact(
                interval=widgets.FloatRangeSlider(
                    value=[v1, v2],
                    min=v1,
                    max=v2,
                    step=0.5,
                    description='Range:',
                    disabled=False,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format='.1f',
                    style={'description_width': 'initial'},
                    layout=Layout(width="50%", height='40px')))
            def zoom_on_data(interval):
                # Get interval
                self.norm_df = self.used_df[
                    (self.used_df[x] > interval[0]) & (
                        self.used_df[x] < interval[1])
                ]
                norm_y = self.norm_df.intensity.mean()

                # Create sources
                source = ColumnDataSource(
                    data=dict(
                        x=self.used_df[x],
                        y=self.used_df[y],
                    ))

                source_interval = ColumnDataSource(
                    data=dict(
                        x=self.norm_df[x],
                        y=self.norm_df[y],
                    ))

                # Create figure
                TOOLTIPS = [
                    (f"{x} (eV), {y}", "($x, $y)"),
                    ("index", "$index"),
                ]

                p = figure(
                    height=self.figure_height, width=self.figure_width,
                    tools="pan, wheel_zoom, box_zoom, reset, undo, redo, crosshair, hover, save",
                    tooltips=TOOLTIPS,
                    active_scroll="wheel_zoom",
                    x_axis_location="above",
                    active_drag="pan",
                    x_axis_label=x + "(eV)",
                    y_axis_label=y,
                )

                # Add line
                p.line("x", "y", source=source,
                       legend_label="Data",
                       color=self.matplotlib_colours[0], line_alpha=0.8,
                       hover_line_alpha=1.0, hover_line_width=2.0)

                p.line("x", "y", source=source_interval,
                       legend_label="Selected data range",
                       color=self.matplotlib_colours[1], line_alpha=0.8,
                       hover_line_alpha=1.0, hover_line_width=2.0)

                # Hide plot by clicking on legend
                p.legend.click_policy = "mute"

                # Show figure
                show(p)

                @interact(
                    norm_data_button=widgets.Checkbox(
                        value=False,
                        description="Normalize data.",
                        indent=False,
                        icon="check"))
                def norm_data(norm_data_button):
                    "Normalize data"
                    if norm_data_button:
                        clear_output(True)

                        # Save normalization
                        setattr(
                            self.used_dataset,
                            self.used_df_name[:-3]+"_norm_y",
                            norm_y),
                        setattr(
                            self.used_dataset,
                            self.used_df_name[:-3]+"_norm_range",
                            interval),
                        self.used_df.intensity = self.used_df.intensity / norm_y

                        # Save as csv
                        self.save_df_to_csv(
                            self.used_dataset, self.used_df_name)

                        # Pickle dataset
                        self.used_dataset.pickle_dataset(
                            self.folders[1] +
                            self.used_dataset.filename.split(".")[0]+".pickle"
                        )

                        print("Normalized DataFrame, and saved as csv.")

        except (AttributeError, KeyError):
            print(f"Wrong Dataset and column combination !")

    def correction_deglitching(self, used_dataset, df, pts, x, y, order):
        """
        Allows one to delete some to replace glitches in the data by using
        linear, square or cubic interpolation.

        :param used_dataset: Dataset used
        :param df: DataFrame used
        :param pts: extra points outside range for data interpolation
        :param x: x axis
        :param y: y axis
        :param order: order of polynom for interpolation
        """
        try:
            self.used_dataset = used_dataset
            self.used_df_name = df
            used_df = getattr(self.used_dataset, self.used_df_name)

            @interact(
                interval=widgets.IntRangeSlider(
                    value=[len(used_df[x]) // 4, len(used_df[x]) // 2],
                    min=pts,
                    max=len(used_df[x]) - 1 - pts,
                    step=1,
                    description='Range (indices):',
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format="d",
                    style={'description_width': 'initial'},
                    layout=Layout(width="50%", height='40px')))
            def deglitch(interval):
                try:
                    plt.close()

                    # Assign values
                    energy, mu = used_df[x], used_df[y]
                    v1, v2 = interval

                    energy_range_1 = energy[v1 - pts:v1]
                    energy_range_2 = energy[v2:v2 + pts]
                    intensity_range_1 = mu[v1 - pts:v1]
                    intensity_range_2 = mu[v2:v2 + pts]

                    # Plot
                    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
                    axs[0].set_title('Raw Data', fontsize=20)
                    axs[0].set_xlabel("Energy", fontsize=15)
                    axs[0].set_ylabel('Intensity', fontsize=15)

                    axs[0].plot(energy, mu, label='Data')
                    axs[0].plot(energy[v1:v2], mu[v1:v2], '-o',
                                linewidth=0.2, label='Selected region')

                    axs[0].axvline(x=energy[v1], color='black', linestyle='--')
                    axs[0].axvline(x=energy[v2], color='black', linestyle='--')
                    axs[0].legend()

                    axs[1].set_title('Zoom', fontsize=20)
                    axs[1].set_xlabel("Energy", fontsize=15)
                    axs[1].set_ylabel('Intensity', fontsize=15)
                    axs[1].tick_params(direction='in', labelsize=15, width=2)

                    axs[1].plot(energy[v1:v2], mu[v1:v2], 'o', color='orange')
                    axs[1].plot(energy_range_1, intensity_range_1,
                                '-o', color='C0')
                    axs[1].plot(energy_range_2, intensity_range_2,
                                '-o', color='C0')

                    axs[1].yaxis.set_label_position("right")
                    axs[1].yaxis.tick_right()

                    # Interpolate
                    energy_range = np.concatenate(
                        (energy_range_1, energy_range_2),
                        axis=0
                    )
                    intensity_range = np.concatenate(
                        (intensity_range_1, intensity_range_2),
                        axis=0
                    )

                    Enew = energy[v1:v2]
                    f1 = interpolate.interp1d(
                        energy_range, intensity_range, kind=order)
                    ITN = f1(Enew)

                    axs[1].plot(Enew, ITN, '--', color='green',
                                label='New line')
                    axs[1].legend()

                    ButtonDeglitch = widgets.Button(
                        description="Interpolate data",
                        layout=Layout(width='25%', height='35px'))
                    display(ButtonDeglitch)

                    @ButtonDeglitch.on_click
                    def ActionButtonDeglitch(selfbutton):
                        used_df = getattr(self.used_dataset, df)
                        used_df[y][v1:v2] = ITN
                        clear_output(True)
                        print(
                            f"Degliched {self.used_dataset.filename}"
                            f"Energy Range: [{energy[v1]}, {energy[v2]}] (eV)")
                        deglitch(interval)
                except ValueError:
                    plt.close()
                    clear_output(True)
                    print(
                        "For quadratic and cubic methods, you need to select more than one extra point.")

        except (AttributeError, KeyError):
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")

    def errors_extraction(
        self,
        used_dataset,
        df,
        x_axis,
        y_axis,
        nbpts,
        deg,
        direction,
        compute
    ):
        """
        """

        def poly(x, y, deg):
            coef = np.polyfit(x, y, deg)
            # Create the polynomial function from the coefficients
            return np.poly1d(coef)(x)

        if compute:
            try:
                clear_output(True)
                self.used_dataset, self.used_df_name = used_dataset, df
                used_df = getattr(self.used_dataset, self.used_df_name)
                x = used_df[x_axis]
                y = used_df[y_axis]

                if nbpts % 2 == 0:
                    n = int(nbpts / 2)
                    self.intl = [k - n if k - n >
                                 0 else 0 for k in range(len(x))]
                    self.intr = [k + n if k + n <
                                 len(x) else len(x) for k in range(len(x))]

                    # Cut the Intervals
                    self.xcut = {f"Int {n}": x.values[i: j] for i, j, n in zip(
                        self.intl, self.intr, range(len(x)))}
                    self.ycut = {f"Int {n}": y.values[i: j] for i, j, n in zip(
                        self.intl, self.intr, range(len(x)))}

                elif nbpts % 2 == 1 and direction == "left":
                    n = int(nbpts / 2)
                    self.intl = [k - n - 1 if k - n - 1 >
                                 0 else 0 for k in range(len(x))]
                    self.intr = [k + n if k + n <
                                 len(x) else len(x) for k in range(len(x))]

                    # Cut the Intervals
                    self.xcut = {f"Int {n}": x.values[i: j] for i, j, n in zip(
                        self.intl, self.intr, range(len(x)))}
                    self.ycut = {f"Int {n}": y.values[i: j] for i, j, n in zip(
                        self.intl, self.intr, range(len(x)))}

                elif nbpts % 2 == 1 and direction == "right":
                    n = int(nbpts / 2)
                    self.intl = [k - n if k - n >
                                 0 else 0 for k in range(len(x))]
                    self.intr = [k + n + 1 if k + n <
                                 len(x) else len(x) for k in range(len(x))]

                    # Cut the Intervals
                    self.xcut = {f"Int {n}": x.values[i: j] for i, j, n in zip(
                        self.ntl, self.intr, range(len(x)))}
                    self.ycut = {f"Int {n}": y.values[i: j] for i, j, n in zip(
                        self.intl, self.intr, range(len(x)))}

                # Create a dictionnary with all the polynoms
                self.polynoms = {f"Poly {i}": poly(
                    self.xcut[f"Int {i}"], self.ycut[f"Int {i}"], deg) for i in range(len(x))}

                self.errors = pd.DataFrame({
                    "Deviations": [self.polynoms[f"Poly {i}"] - self.ycut[f"Int {i}"] for i in range(len(x))],
                    "RMS": [(np.sum((self.polynoms[f"Poly {i}"] - self.ycut[f"Int {i}"]) ** 2)/len(self.ycut[f"Int {i}"])) ** (1 / 2) for i in range(len(x))]
                })

                print("RMS and deviation well determined.")

                plt.close()

                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                axs[0].set_xlabel("Energy")
                axs[0].set_ylabel('Intensity')
                axs[0].set_title('Data')
                axs[0].plot(x, y, label="Data")
                axs[0].legend()

                axs[1].set_xlabel("Energy")
                axs[1].set_ylabel('Intensity')
                axs[1].set_title('Root Mean Square')
                axs[1].plot(x, self.errors["RMS"], label="RMS")
                axs[1].legend()

                try:
                    used_df["Deviations"] = self.errors["Deviations"]
                    used_df["RMS"] = self.errors["RMS"]
                except:
                    setattr(self.used_dataset, self.used_df_name, pd.concat(
                        [used_df, self.errors], axis=1, sort=False))

                display(getattr(self.used_dataset, self.used_df_name))

            except (AttributeError, KeyError):
                plt.close()
                if y_axis == "value":
                    print("Please select a column.")
                else:
                    print(f"Wrong Dataset and column combination !")

            except Exception as e:
                raise e

        else:
            plt.close()
            print("Window cleared.")
            clear_output(True)

    def LCF(
        self,
        ref_spectra,
        spec_number,
        used_dataset,
        df_type,
        x,
        y,
        LCF_bool
    ):
        """
        """
        if LCF_bool and len(ref_spectra) > 1:
            self.ref_names = [f.filename for f in ref_spectra]

            try:
                def align_ref_and_spec(**kwargs):
                    try:
                        # interval for data
                        v1Data, v2Data = [], []
                        for j, C in enumerate(spec_number):
                            used_df = getattr(C, df_type)
                            try:
                                v1Data.append(
                                    int(np.where(
                                        used_df["Energy"].values ==
                                        self.energy_widgets[0].value[0])[0]))
                            except TypeError:
                                v1Data.append(0)

                            try:
                                v2Data.append(
                                    int(np.where(
                                        used_df["Energy"].values ==
                                        self.energy_widgets[0].value[1])[0]))
                            except TypeError:
                                v2Data.append(len(used_df["Energy"].values)-1)

                        # Take data spectrum on interval
                        self.spec_df = [getattr(D, df_type).copy(
                        )[v1:v2] for D, v1, v2 in zip(spec_number, v1Data, v2Data)]

                        self.used_df = self.spec_df[spec_number.index(
                            used_dataset)]

                        # Import the references
                        self.ref_df = [getattr(f, df_type).copy()
                                       for f in ref_spectra]

                        # Add shifts and scale factors to references
                        shifts = [c.value for c in shift_widgets]
                        factors = [
                            c.value for c in self.intensity_factor_widgets]

                        for df, s, a in zip(self.ref_df, shifts, factors):
                            df[x] = np.round(df[x].values + s, 2)
                            df[y] = df[y]*a

                        # interval for references after corrections
                        v1Ref, v2Ref = [], []
                        for used_df in self.ref_df:
                            try:
                                v1Ref.append(
                                    int(np.where(
                                        used_df["Energy"].values ==
                                        self.energy_widgets[0].value[0])[0]))
                            except TypeError:
                                v1Ref.append(0)

                            try:
                                v2Ref.append(
                                    int(np.where(
                                        used_df["Energy"].values ==
                                        self.energy_widgets[0].value[1])[0]))
                            except TypeError:
                                v2Ref.append(len(used_df["Energy"].values)-1)

                        # Take ref on interval
                        self.ref_df = [df[v1:v2] for df, v1,
                                       v2 in zip(self.ref_df, v1Ref, v2Ref)]

                        # Plotting
                        fig, ax = plt.subplots(figsize=(16, 6))
                        for used_df, n in zip(self.ref_df, self.ref_names):
                            ax.plot(used_df[x], used_df[y],
                                    "--", label=f"reference {n}")

                        ax.plot(
                            self.used_df[x], self.used_df[y], label=used_dataset.filename)
                        ax.legend()
                        plt.title("First visualization of the data")
                        plt.show()

                        # Check if all the references and data spectra have the same interval and nb of points
                        good_range_ref = [np.array_equal(
                            df[x].values, self.used_df[x].values) for df in self.ref_df]
                        good_range_spec = [np.array_equal(
                            df[x].values, self.used_df[x].values) for df in self.spec_df]

                        if all(good_range_ref) and all(good_range_spec):
                            print("""The energy ranges between the references and the data match.\nYou should shift the references so that the features are aligned in energy, and scale them so that the weight of each reference during the Linear Combination Fit is lower than 1.""")

                            ButtonLauchLCF = Button(
                                description="Launch LCF",
                                layout=Layout(width='40%', height='35px'))
                            display(ButtonLauchLCF)

                            @ButtonLauchLCF.on_click
                            def ActionLauchLCF(selfbutton):

                                for used_df, C in zip(self.spec_df, spec_number):

                                    # Create function that returns the square of the difference between LCf of references and data, for each Dataset that was selected
                                    def ref_model(pars):
                                        # Sum weighted ref, initialized
                                        y_sum = np.zeros(
                                            len(self.ref_df[0].values))

                                        # fig, ax = plt.subplots(figsize = (16, 6))

                                        # All ref but last one
                                        for used_df, p in zip(self.ref_df[:-1], pars):
                                            y_sum += used_df[y].values * p

                                            # print(p)
                                            # ax.plot(used_df[y].values * p, label = n)

                                        # Last ref
                                        y_sum += self.ref_df[-1][y].values * \
                                            (1 - np.sum(pars))

                                        # print(1-np.sum(pars))
                                        # ax.plot(self.ref_df[-1][y].values * (1-np.sum(pars)), label = self.ref_names[-1])

                                        # Data
                                        # ax.plot(self.InterpolatedUsedDf[y].values, label ="Data")
                                        # ax.legend()

                                        return np.sum((used_df[y].values - y_sum) ** 2)

                                    # Launch fit
                                    initial_guess = np.ones(
                                        len(self.ref_df)-1) / len(self.ref_df)
                                    print(initial_guess)
                                    boundaries = [(0, 1) for i in range(
                                        len(self.ref_names)-1)]

                                    LCF_result = optimize.minimize(
                                        ref_model, initial_guess, bounds=boundaries, method='TNC')
                                    setattr(C, "LCF_result", LCF_result)

                                    ref_weights = np.append(
                                        LCF_result.x.copy(), 1 - np.sum(LCF_result.x))
                                    setattr(C, "ref_weights", ref_weights)

                                    # Plotting result
                                    fig, ax = plt.subplots(figsize=(16, 6))
                                    for data, n, w in zip(self.ref_df, self.ref_names, ref_weights):
                                        ax.plot(
                                            data[x], data[y] * w, "--", label=f"{n} component * {w}")
                                        used_df[f"WeighedInterpolatedIt_{n}"] = data[y]*w

                                    sum_ref_weights = np.sum(
                                        [df[y] * w for df, w in zip(self.ref_df, ref_weights)], axis=0)

                                    used_df["sum_ref_weights"] = sum_ref_weights
                                    setattr(C, "LCF_df", used_df)

                                    ax.plot(
                                        used_df[x], sum_ref_weights, label="Sum of weighted references.")
                                    ax.plot(
                                        used_df[x], used_df[y], label=C.filename)
                                    ax.legend()
                                    plt.title("LCF Result")
                                    plt.show()

                                    # Print detailed result
                                    print(
                                        f"The detail of the fitting for {C.filename} is the following:")
                                    print(LCF_result)
                                    print(
                                        f"The weights for the references are {ref_weights}")

                                    r_factor = np.sum(
                                        (self.used_df[y]-sum_ref_weights)**2) / np.sum((self.used_df[y])**2)
                                    print(f"R factor :{r_factor}")
                                    setattr(C, "ref_R_factor", r_factor)

                                # Final plot of the reference weights
                                fig, ax = plt.subplots(figsize=(10, 6))

                                xplot = np.arange(0, len(spec_number), 1)
                                for j, n in enumerate(self.ref_names):
                                    ax.plot(xplot, [
                                            C.ref_weights[j] for C in spec_number], marker='x', label=f"{n} component")
                                ax.legend()
                                ax.set_ylabel('Weight')
                                ax.set_title(
                                    'Comparison of weight importance thoughout data series')
                                ax.set_xticks(xplot)
                                ax.set_xticklabels(
                                    [C.filename for C in spec_number], rotation=90, fontsize=14)

                        else:
                            print(
                                "The energy ranges between the references and the data do not match.")

                    except (AttributeError, KeyError):
                        print(f"Wrong Dataset and column combination !")

                    # except(ValueError):
                    #     print("Select at least two refs and one spectrum.")

                    except Exception as e:
                        raise e

                # shift the references
                shift_widgets = [widgets.FloatText(
                    value=0,
                    step=self.interpol_step,
                    continuous_update=False,
                    readout=True,
                    readout_format='.2f',
                    description=f"Shift for {n}",
                    style={'description_width': 'initial'}) for n in self.ref_names]
                _list_shift_widgets = widgets.HBox(tuple(shift_widgets))

                # Multiply the reference intensity
                self.intensity_factor_widgets = [widgets.FloatText(
                    value=1,
                    continuous_update=False,
                    readout=True,
                    readout_format='.2f',
                    description=f"Intensity Factor for {n}",
                    style={'description_width': 'initial'}) for n in self.ref_names]
                _list_intensity_factor_widgets = widgets.HBox(
                    tuple(self.intensity_factor_widgets))

                # Select the energy range of interest for the data
                self.energy_widgets = [widgets.FloatRangeSlider(
                    min=self.new_energy_column[0],
                    value=[self.new_energy_column[0],
                           self.new_energy_column[-1]],
                    max=self.new_energy_column[-1],
                    step=self.interpol_step,
                    description='Energy range (eV):',
                    disabled=False,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format='.2f',
                    style={'description_width': 'initial'},
                    layout=Layout(width="50%", height='40px'))]
                _list_energy_widgets = widgets.HBox(tuple(self.energy_widgets))

                align_ref_and_spec_dict = {
                    c.description: c for c in self.energy_widgets + shift_widgets + self.intensity_factor_widgets}

                _list_in = widgets.VBox(
                    [_list_energy_widgets, _list_shift_widgets, _list_intensity_factor_widgets])
                _list_out = widgets.interactive_output(
                    align_ref_and_spec, align_ref_and_spec_dict)
                display(_list_in, _list_out)

            except (AttributeError, KeyError):
                print(f"Wrong Dataset and column combination !")

            except Exception as e:
                raise e
        else:
            clear_output(True)
            print("Select at least two references, one Dataset that will serve to visualise the interpolation, and the list of datasets you would like to process.")

    # Reduction interactive function
    def reduce_data(
        self,
        method,
        used_class_list,
        used_dataset,
        df,
        plot_bool
    ):
        """
        Define the reduction routine to follow depending on the
        Reduction widget state.

        :param method:
        :param used_class_list:
        :param used_dataset:
        :param df:
        :param plot_bool:
        """
        try:
            self.used_class_list = used_class_list
            self.used_dataset = used_dataset
            self.used_dataset_position = used_class_list.index(used_dataset)
            clear_output(True)

            # Update
            self._list_reduce_LSF.children[0].value = "value"
            self._list_reduce_chebyshev.children[0].value = "value"
            self._list_reduce_polynoms.children[0].value = "value"
            self._list_reduce_splines_derivative.children[0].value = "value"

            try:
                self.used_df_name = df
                used_df = getattr(self.used_dataset, self.used_df_name)

            except (AttributeError, KeyError):
                print(f"Wrong Dataset and column combination !")

            if method == "LSF" and plot_bool:
                display(self.widget_list_reduce_LSF)
            if method == "Chebyshev" and plot_bool:
                display(self.widget_list_reduce_chebyshev)
            if method == "Polynoms" and plot_bool:
                display(self.widget_list_reduce_polynoms)
            if method == "SingleSpline" and plot_bool:
                display(self.widget_list_reduce_single_spline)
            if method == "Splines" and plot_bool:
                self._list_reduce_splines_derivative.children[0].disabled = False
                self._list_reduce_splines_derivative.children[1].disabled = False
                display(self.widget_list_reduce_splines_derivative)
            if method == "NormMax" and plot_bool:
                display(self.widget_list_normalize_maxima)
            if not plot_bool:
                print("Window cleared")
                plt.close()

        except ValueError:
            clear_output(True)
            print(
                f"{used_dataset.filename} is not in the list of datasets to reduce.")

    # Reduction interactive sub-functions
    def reduce_LSF(self, y, interval, lam, p):
        """Reduce the background following a Least Square Fit method"""

        def baseline_als(y, lam, p, niter=10):
            """Polynomial function defined by sparse"""
            L = len(y)
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
            w = np.ones(L)
            for i in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.dot(D.transpose())
                z = sparse.linalg.spsolve(Z, w * y)
                w = p * (y > z) + (1 - p) * (y < z)
            return z

        try:
            number = self.used_dataset_position
            df = self.used_df_name

            # Retrieve original data
            mu, energy, v1, v2 = [], [], [], []
            for j, C in enumerate(self.used_class_list):
                used_df = getattr(C, df)
                mu.append(used_df[y].values)
                energy.append(used_df["Energy"].values)

                try:
                    v1.append(int(np.where(energy[j] == interval[0])[0]))
                except TypeError:
                    v1.append(0)

                try:
                    v2.append(int(np.where(energy[j] == interval[1])[0]))
                except TypeError:
                    v2.append(len(energy[j])-1)

            ButtonRemoveBackground = Button(
                description="Remove background for all",
                layout=Layout(width='30%', height='35px'))
            ButtonSaveDataset = Button(
                description="Save reduced data for this Dataset",
                layout=Layout(width='30%', height='35px'))
            display(widgets.HBox((ButtonRemoveBackground, ButtonSaveDataset)))

            plt.close()

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
            axs[0].set_xlabel("Energy")
            axs[0].set_ylabel('Intensity')
            axs[0].set_title('Raw Data')
            axs[0].tick_params(direction='in', labelsize=15, width=2)

            # Compute from sliders
            baseline = baseline_als(
                mu[number][v1[number]:v2[number]], lam, p * 10 ** (-3), niter=10)

            axs[0].plot(energy[number], mu[number], label='Data')
            axs[0].plot(energy[number][v1[number]:v2[number]], mu[number]
                        [v1[number]:v2[number]], '-o', label='Selected Region')
            axs[0].plot(energy[number][v1[number]:v2[number]],
                        baseline, '--', color='green', label='Bkg')
            axs[0].axvline(x=energy[number][v1[number]],
                           color='black', linestyle='--')
            axs[0].axvline(x=energy[number][v2[number]],
                           color='black', linestyle='--')
            axs[0].legend()

            difference = mu[number][v1[number]:v2[number]] - baseline

            axs[1].set_title('Background subtracted')
            axs[1].set_xlabel("Energy")
            axs[1].set_ylabel('Intensity')
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right()
            axs[1].tick_params(direction='in', labelsize=15, width=2)
            axs[1].set_xlim(energy[number][v1[number]],
                            energy[number][v2[number]])

            axs[1].plot(energy[number][v1[number]:v2[number]],
                        difference, '-', color='C0')

            print("Channel 1:", v1[number], ";",
                  "energy:", energy[number][v1[number]])
            print("Channel 2:", v2[number], ";",
                  "energy:", energy[number][v2[number]])

            @ButtonSaveDataset.on_click
            def ActionSaveDataset(selfbutton):
                # Save single Dataset without background in Class
                C = self.used_dataset
                IN = mu[number][v1[number]:v2[number]] / \
                    np.trapz(
                        difference, x=energy[number][v1[number]:v2[number]])
                temp_df = pd.DataFrame()
                temp_df["Energy"] = energy[number][v1[number]:v2[number]]
                temp_df["\u03BC"] = mu[number][v1[number]:v2[number]]
                temp_df["background_corrected"] = difference
                temp_df["\u03BC_variance"] = [
                    1 / d if d > 0 else 0 for d in difference]
                temp_df["second_normalized_\u03BC"] = IN
                display(temp_df)
                setattr(C, "reduced_df", temp_df)
                print(f"Saved Dataset {C.filename}")
                temp_df.to_csv(
                    f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

            @ButtonRemoveBackground.on_click
            def ActionRemoveBackground(selfbutton):
                # Substract background to the intensity
                clear_output(True)

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                ax[0].set_title('Background subtracted')
                ax[0].set_xlabel("Energy")
                ax[0].set_ylabel('Intensity')
                ax[0].tick_params(direction='in', labelsize=15, width=2)

                ax[1].set_title('Background subtracted shifted')
                ax[1].set_xlabel("Energy")
                ax[1].set_ylabel('Intensity')
                ax[1].yaxis.tick_right()
                ax[1].tick_params(direction='in', labelsize=15, width=2)
                ax[1].yaxis.set_label_position("right")

                ITB = []
                try:
                    for i in range(len(mu)):
                        baseline = baseline_als(
                            mu[i][v1[i]:v2[i]], lam, p * 10 ** (-3), niter=10)
                        ITnew = mu[i][v1[i]:v2[i]] - baseline
                        ITB.append(ITnew)

                    for i in range(len(ITB)):
                        ax[0].plot(energy[i][v1[i]:v2[i]], ITB[i])
                        ax[1].plot(energy[i][v1[i]:v2[i]], ITB[i] +
                                   0.1*(i), label=self.used_class_list[i].filename)

                    ax[1].legend(loc='upper center', bbox_to_anchor=(
                        0, -0.2), fancybox=True, shadow=True, ncol=5)

                except Exception as e:
                    print(e)

                ButtonSave = widgets.Button(
                    description="Save background reduced data",
                    layout=Layout(width='30%', height='35px'))
                ButtonNormalize = widgets.Button(
                    description='Normalize for all',
                    layout=Layout(width='30%', height='35px'))
                display(widgets.HBox((ButtonNormalize, ButtonSave)))

                @ButtonSave.on_click
                def ActionButtonSave(selfbutton):
                    # Save intensity without background
                    for j, C in enumerate(self.used_class_list):
                        temp_df = pd.DataFrame()
                        temp_df = getattr(C, "reduced_df")
                        temp_df["Energy"] = energy[j][v1[j]:v2[j]]
                        temp_df["\u03BC"] = mu[j][v1[j]:v2[j]]
                        temp_df["background_corrected"] = ITB[j]
                        temp_df["\u03BC_variance"] = [
                            1 / d if d > 0 else 0 for d in ITB[j]]
                        setattr(C, "reduced_df", temp_df)
                        print(f"Saved Dataset {C.filename}")
                        temp_df.to_csv(
                            f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

                @ButtonNormalize.on_click
                def ActionButtonNormalize(selfbutton):
                    # Normalize data

                    clear_output(True)
                    area = []
                    ITN = []
                    for i in range(len(ITB)):
                        areaV = np.trapz(ITB[i], x=energy[i][v1[i]:v2[i]])
                        area.append(areaV)
                        ITN.append(ITB[i]/area[i])

                    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                    ax[0].set_title('Background subtracted normalized')
                    ax[0].set_xlabel("Energy")
                    ax[0].set_ylabel('Intensity')
                    ax[0].tick_params(direction='in', labelsize=15, width=2)
                    ax[1].set_title(
                        'Background subtracted normalized & shifted')
                    ax[1].set_xlabel("Energy")
                    ax[1].set_ylabel('Intensity')
                    ax[1].yaxis.set_label_position("right")
                    ax[1].yaxis.tick_right()
                    ax[1].tick_params(direction='in', labelsize=15, width=2)

                    for i in range(len(ITN)):
                        ax[0].plot(energy[i][v1[i]:v2[i]], ITN[i])
                        ax[1].plot(energy[i][v1[i]:v2[i]], ITN[i] +
                                   0.1*(i+1), label=self.used_class_list[i].filename)

                    ax[1].legend(loc='upper center', bbox_to_anchor=(
                        0, -0.2), fancybox=True, shadow=True, ncol=5)

                    ButtonSaveNormalizedData = widgets.Button(
                        description="Save normalized data",
                        layout=Layout(width='30%', height='35px'))
                    display(ButtonSaveNormalizedData)

                    @ButtonSaveNormalizedData.on_click
                    def ActionSaveNormalizedData(selfbutton):
                        # Save normalized data
                        for j, C in enumerate(self.used_class_list):
                            temp_df = pd.DataFrame()
                            temp_df["Energy"] = energy[j][v1[j]:v2[j]]
                            temp_df["\u03BC"] = mu[j][v1[j]:v2[j]]
                            temp_df["background_corrected"] = ITB[j]
                            temp_df["\u03BC_variance"] = [
                                1 / d if d > 0 else 0 for d in ITB[j]]
                            temp_df["second_normalized_\u03BC"] = ITN[j]
                            setattr(C, "reduced_df", temp_df)
                            print(f"Saved Dataset {C.filename}")
                            temp_df.to_csv(
                                f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

        except (AttributeError, KeyError):
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")

        except (ValueError, NameError):
            plt.close()
            print("The selected energy range is wrong.")

    def reduce_chebyshev(self, y, interval, p, n):
        """Reduce the background with chebyshev polynomails"""

        def chebyshev(x, y, d, n):
            """
            Define a chebyshev polynomial using np.polynomial.chebyshev.fit
            method
            """
            w = (1/y) ** n
            p = np.polynomial.Chebyshev.fit(x, y, d, w=w)

            return p(x)

        try:
            number = self.used_dataset_position
            df = self.used_df_name

            # Retrieve original data
            mu, energy, v1, v2 = [], [], [], []
            for j, C in enumerate(self.used_class_list):
                used_df = getattr(C, df)
                mu.append(used_df[y].values)
                energy.append(used_df["Energy"].values)

                try:
                    v1.append(int(np.where(energy[j] == interval[0])[0]))
                except TypeError:
                    v1.append(0)

                try:
                    v2.append(int(np.where(energy[j] == interval[1])[0]))
                except TypeError:
                    v2.append(len(energy[j])-1)

            plt.close()

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
            axs[0].set_xlabel("Energy")
            axs[0].set_ylabel('Intensity')
            axs[0].set_title('Raw Data')
            axs[0].tick_params(direction='in', labelsize=15, width=2)

            # Compute from sliders
            baseline = chebyshev(
                energy[number][v1[number]:v2[number]], mu[number][v1[number]:v2[number]], p, n)

            axs[0].plot(energy[number], mu[number], label='Data')
            axs[0].plot(energy[number][v1[number]:v2[number]], mu[number]
                        [v1[number]:v2[number]], '-o', label='Selected Region')
            axs[0].plot(energy[number][v1[number]:v2[number]],
                        baseline, '--', color='green', label='Bkg')
            axs[0].axvline(x=energy[number][v1[number]],
                           color='black', linestyle='--')
            axs[0].axvline(x=energy[number][v2[number]],
                           color='black', linestyle='--')
            axs[0].legend()

            difference = mu[number][v1[number]:v2[number]] - baseline

            axs[1].set_title('Background subtracted')
            axs[1].set_xlabel("Energy")
            axs[1].set_ylabel('Intensity')
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right()
            axs[1].tick_params(direction='in', labelsize=15, width=2)
            axs[1].set_xlim(energy[number][v1[number]],
                            energy[number][v2[number]])

            axs[1].plot(energy[number][v1[number]:v2[number]],
                        difference, '-', color='C0')

            print("Channel 1:", v1[number], ";",
                  "energy:", energy[number][v1[number]])
            print("Channel 2:", v2[number], ";",
                  "energy:", energy[number][v2[number]])

            ButtonRemoveBackground = Button(
                description="Remove background for all",
                layout=Layout(width='30%', height='35px'))
            ButtonSaveDataset = Button(
                description="Save reduced data for this Dataset",
                layout=Layout(width='30%', height='35px'))
            display(widgets.HBox((ButtonRemoveBackground, ButtonSaveDataset)))

            @ButtonSaveDataset.on_click
            def ActionSaveDataset(selfbutton):
                # Save single Dataset without background in Class
                C = self.used_dataset
                IN = mu[number][v1[number]:v2[number]] / \
                    np.trapz(
                        difference, x=energy[number][v1[number]:v2[number]])
                temp_df = pd.DataFrame()
                temp_df["Energy"] = energy[number][v1[number]:v2[number]]
                temp_df["\u03BC"] = mu[number][v1[number]:v2[number]]
                temp_df["background_corrected"] = difference
                temp_df["\u03BC_variance"] = [
                    1 / d if d > 0 else 0 for d in difference]
                temp_df["second_normalized_\u03BC"] = IN
                display(temp_df)
                setattr(C, "reduced_df", temp_df)
                print(f"Saved Dataset {C.filename}")
                temp_df.to_csv(
                    f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

            @ButtonRemoveBackground.on_click
            def ActionRemoveBackground(selfbutton):
                # Substract background to the intensity
                clear_output(True)

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                ax[0].set_title('Background subtracted')
                ax[0].set_xlabel("Energy")
                ax[0].set_ylabel('Intensity')
                ax[0].tick_params(direction='in', labelsize=15, width=2)
                ax[0].set_xlim(energy[number][v1[number]],
                               energy[number][v2[number]])

                ax[1].set_title('Background subtracted shifted')
                ax[1].set_xlabel("Energy")
                ax[1].set_ylabel('Intensity')
                ax[1].yaxis.tick_right()
                ax[1].tick_params(direction='in', labelsize=15, width=2)
                ax[1].set_xlim(energy[number][v1[number]],
                               energy[number][v2[number]])
                ax[1].yaxis.set_label_position("right")
                # ax.plot(energy[v1:v2],ITN)

                ITB = []
                try:
                    for i in range(len(mu)):
                        baseline = chebyshev(
                            energy[i][v1[i]:v2[i]], mu[i][v1[i]:v2[i]], p, n)
                        ITnew = mu[i][v1[i]:v2[i]]-baseline
                        ITB.append(ITnew)

                    for i in range(len(ITB)):
                        ax[0].plot(energy[i][v1[i]:v2[i]], ITB[i])
                        ax[1].plot(energy[i][v1[i]:v2[i]], ITB[i] +
                                   0.1*(i), label=self.used_class_list[i].filename)

                    ax[1].legend(loc='upper center', bbox_to_anchor=(
                        0, -0.2), fancybox=True, shadow=True, ncol=5)

                except ValueError:
                    print("The energy range is wrong.")

                ButtonSave = widgets.Button(
                    description="Save background reduced data",
                    layout=Layout(width='30%', height='35px'))
                ButtonNormalize = widgets.Button(
                    description='Normalize for all',
                    layout=Layout(width='30%', height='35px'))
                display(widgets.HBox((ButtonNormalize, ButtonSave)))

                @ButtonSave.on_click
                def ActionButtonSave(selfbutton):
                    # Save intensity without background
                    for j, C in enumerate(self.used_class_list):
                        temp_df = pd.DataFrame()
                        temp_df["Energy"] = energy[j][v1[j]:v2[j]]
                        temp_df["\u03BC"] = mu[j][v1[j]:v2[j]]
                        temp_df["background_corrected"] = ITB[j]
                        temp_df["\u03BC_variance"] = [
                            1/d if d > 0 else 0 for d in ITB[j]]
                        setattr(C, "reduced_df", temp_df)
                        print(f"Saved Dataset {C.filename}")
                        temp_df.to_csv(
                            f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

                @ButtonNormalize.on_click
                def ActionButtonNormalize(selfbutton):
                    # Normalize data

                    clear_output(True)
                    area = []
                    ITN = []
                    for i in range(len(ITB)):
                        areaV = np.trapz(ITB[i], x=energy[i][v1[i]:v2[i]])
                        area.append(areaV)
                        ITN.append(ITB[i]/area[i])

                    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                    ax[0].set_title('Background subtracted normalized')
                    ax[0].set_xlabel("Energy")
                    ax[0].set_ylabel('Intensity')
                    ax[0].tick_params(direction='in', labelsize=15, width=2)
                    ax[0].set_xlim(energy[number][v1[number]],
                                   energy[number][v2[number]])
                    ax[1].set_title(
                        'Background subtracted normalized & shifted')
                    ax[1].set_xlabel("Energy")
                    ax[1].set_ylabel('Intensity')
                    ax[1].yaxis.set_label_position("right")
                    ax[1].yaxis.tick_right()
                    ax[1].tick_params(direction='in', labelsize=15, width=2)
                    ax[1].set_xlim(energy[number][v1[number]],
                                   energy[number][v2[number]])

                    for i in range(len(ITN)):
                        ax[0].plot(energy[i][v1[i]:v2[i]], ITN[i])
                        ax[1].plot(energy[i][v1[i]:v2[i]], ITN[i] + 0.1 *
                                   (i+1), label=self.used_class_list[i].filename)

                    ax[1].legend(loc='upper center', bbox_to_anchor=(
                        0, -0.2), fancybox=True, shadow=True, ncol=5)

                    ButtonSaveNormalizedData = widgets.Button(
                        description="Save normalized data",
                        layout=Layout(width='30%', height='35px'))
                    display(ButtonSaveNormalizedData)

                    @ButtonSaveNormalizedData.on_click
                    def ActionSaveNormalizedData(selfbutton):
                        # Save normalized data
                        for j, C in enumerate(self.used_class_list):
                            temp_df = pd.DataFrame()
                            temp_df["Energy"] = energy[j][v1[j]:v2[j]]
                            temp_df["\u03BC"] = mu[j][v1[j]:v2[j]]
                            temp_df["background_corrected"] = ITB[j]
                            temp_df["\u03BC_variance"] = [
                                1/d if d > 0 else 0 for d in ITB[j]]
                            temp_df["second_normalized_\u03BC"] = ITN[j]
                            setattr(C, "reduced_df", temp_df)
                            print(f"Saved Dataset {C.filename}")
                            temp_df.to_csv(
                                f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

        except (AttributeError, KeyError):
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")
        except (ValueError, NameError):
            plt.close()
            print("The selected energy range is wrong.")

    def reduce_polynoms(self, y, interval, sL):
        """
        Reduce the background using a fixed number of points and
        Polynoms between them
        """
        try:
            number = self.used_dataset_position
            df = self.used_df_name

            # Retrieve original data
            mu, energy, v1, v2 = [], [], [], []
            for j, C in enumerate(self.used_class_list):
                used_df = getattr(C, df)
                mu.append(used_df[y].values)
                energy.append(used_df["Energy"].values)

                try:
                    v1.append(int(np.where(energy[j] == interval[0])[0]))
                except TypeError:
                    v1.append(0)

                try:
                    v2.append(int(np.where(energy[j] == interval[1])[0]))
                except TypeError:
                    v2.append(len(energy[j])-1)

            plt.close()

            def plot_sliders(**selfsliders):
                # Take values from sliders
                positions = []
                for i in range(sL):
                    positions.append(controls[i].value)
                positions.sort()

                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                axs[0].set_xlabel("Energy")
                axs[0].set_ylabel('Intensity')
                axs[0].set_title('Raw Data')
                axs[0].tick_params(direction='in', labelsize=15, width=2)

                axs[0].plot(energy[number], mu[number], label='Data')
                axs[0].plot(energy[number][v1[number]:v2[number]], mu[number]
                            [v1[number]:v2[number]], '-o', label='Selected Region')
                axs[0].axvline(x=energy[number][v1[number]],
                               color='black', linestyle='--')
                axs[0].axvline(x=energy[number][v2[number]],
                               color='black', linestyle='--')
                axs[0].set_ylim(np.min(mu[number])-0.01*np.min(mu[number]),
                                np.max(mu[number])+0.01*np.max(mu[number]))

                energy_int = energy[number][positions]
                data_int = mu[number][positions]

                baseline = interpolate.splrep(energy_int, data_int, s=0)
                ITint = interpolate.splev(
                    energy[number][v1[number]:v2[number]], baseline, der=0)
                axs[0].plot(energy[number][v1[number]:v2[number]],
                            ITint, '--', color='green', label='Bkg')

                for i in range(sL):
                    axs[0].plot(energy[number][positions[i]], mu[number]
                                [positions[i]], 'o', color='black', markersize=10)
                    axs[0].axvline(x=energy[number][positions[i]])
                axs[0].legend()
                difference = mu[number][v1[number]:v2[number]] - ITint

                axs[1].plot(energy[number][v1[number]:v2[number]], difference)
                axs[1].set_title('Bgk subtracted')
                axs[1].set_xlabel("Energy")
                axs[1].set_ylabel('Intensity')
                axs[1].yaxis.set_label_position("right")
                axs[1].tick_params(direction='in', labelsize=15, width=2)
                axs[1].set_xlim(energy[number][v1[number]],
                                energy[number][v2[number]])

                #################################### BUTTONS ###############################################################################
                ButtonRemoveBackground = Button(
                    description="Remove background for all",
                    layout=Layout(width='30%', height='35px'))
                ButtonSaveDataset = Button(
                    description="Save reduced data for this Dataset",
                    layout=Layout(width='30%', height='35px'))
                display(widgets.HBox((ButtonRemoveBackground, ButtonSaveDataset)))

                @ButtonSaveDataset.on_click
                def ActionSaveDataset(selfbutton):
                    # Save single Dataset without background in Class
                    C = self.used_dataset
                    IN = mu[number][v1[number]:v2[number]] / \
                        np.trapz(
                            difference, x=energy[number][v1[number]:v2[number]])
                    temp_df = pd.DataFrame()
                    temp_df["Energy"] = energy[number][v1[number]:v2[number]]
                    temp_df["\u03BC"] = mu[number][v1[number]:v2[number]]
                    temp_df["background_corrected"] = difference
                    temp_df["\u03BC_variance"] = [
                        1/d if d > 0 else 0 for d in difference]
                    temp_df["second_normalized_\u03BC"] = IN
                    display(temp_df)
                    setattr(C, "reduced_df", temp_df)
                    print(f"Saved Dataset {C.filename}")
                    temp_df.to_csv(
                        f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

                @ButtonRemoveBackground.on_click
                def ActionRemoveBackground(selfbutton):
                    # Substract background to the intensity
                    ITB = []
                    for i in range(len(mu)):
                        baseline = interpolate.splrep(
                            energy_int, data_int, s=0)
                        ITint = interpolate.splev(
                            energy[i][v1[i]:v2[i]], baseline, der=0)
                        ITnew = mu[i][v1[i]:v2[i]] - ITint
                        ITB.append(ITnew)

                    clear_output(True)

                    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                    ax[0].set_title('Background subtracted')
                    ax[0].set_xlabel("Energy")
                    ax[0].set_ylabel('Intensity')
                    ax[0].tick_params(direction='in', labelsize=15, width=2)
                    ax[0].set_xlim(energy[number][v1[number]],
                                   energy[number][v2[number]])

                    ax[1].set_title('Background subtracted shifted')
                    ax[1].set_xlabel("Energy")
                    ax[1].set_ylabel('Intensity')
                    ax[1].yaxis.tick_right()
                    ax[1].tick_params(direction='in', labelsize=15, width=2)
                    ax[1].set_xlim(energy[number][v1[number]],
                                   energy[number][v2[number]])
                    ax[1].yaxis.set_label_position("right")

                    for i in range(len(ITB)):
                        ax[0].plot(energy[i][v1[i]:v2[i]], ITB[i])
                        ax[1].plot(energy[i][v1[i]:v2[i]], ITB[i] +
                                   0.1 * (i), label=self.used_class_list[i].filename)

                    ax[1].legend(loc='upper center', bbox_to_anchor=(
                        0, -0.2), fancybox=True, shadow=True, ncol=5)

                    ButtonSave = widgets.Button(
                        description="Save background reduced data",
                        layout=Layout(width='30%', height='35px'))
                    ButtonNormalize = widgets.Button(
                        description='Normalize for all',
                        layout=Layout(width='30%', height='35px'))
                    display(widgets.HBox((ButtonNormalize, ButtonSave)))

                    @ButtonSave.on_click
                    def ActionButtonSave(selfbutton):
                        # Save intensity without background
                        for j, C in enumerate(self.used_class_list):
                            temp_df = pd.DataFrame()
                            temp_df["Energy"] = energy[j][v1[j]:v2[j]]
                            temp_df["\u03BC"] = mu[j][v1[j]:v2[j]]
                            temp_df["background_corrected"] = ITB[j]
                            temp_df["\u03BC_variance"] = [
                                1/d if d > 0 else 0 for d in ITB[j]]
                            setattr(C, "reduced_df", temp_df)
                            print(f"Saved Dataset {C.filename}")
                            temp_df.to_csv(
                                f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

                    @ButtonNormalize.on_click
                    def ActionButtonNormalize(selfbutton):
                        # Normalize data
                        clear_output(True)
                        area = []
                        ITN = []
                        for i in range(len(ITB)):
                            areaV = np.trapz(ITB[i], x=energy[i][v1[i]:v2[i]])
                            area.append(areaV)
                            ITN.append(ITB[i]/area[i])

                        fig, ax = plt.subplots(
                            nrows=1, ncols=2, figsize=(16, 6))
                        ax[0].set_title('Background subtracted normalized')
                        ax[0].set_xlabel("Energy")
                        ax[0].set_ylabel('Intensity')
                        ax[0].tick_params(
                            direction='in', labelsize=15, width=2)
                        ax[0].set_xlim(energy[number][v1[number]],
                                       energy[number][v2[number]])
                        ax[1].set_title(
                            'Background subtracted normalized & shifted')
                        ax[1].set_xlabel("Energy")
                        ax[1].set_ylabel('Intensity')
                        ax[1].yaxis.set_label_position("right")
                        ax[1].yaxis.tick_right()
                        ax[1].tick_params(
                            direction='in', labelsize=15, width=2)
                        ax[1].set_xlim(energy[number][v1[number]],
                                       energy[number][v2[number]])
                        for i in range(len(ITN)):
                            ax[0].plot(energy[i][v1[i]:v2[i]], ITN[i])
                            ax[1].plot(energy[i][v1[i]:v2[i]], ITN[i] + 0.1 *
                                       (i+1), label=self.used_class_list[i].filename)

                        ax[1].legend(loc='upper center', bbox_to_anchor=(
                            0, -0.2), fancybox=True, shadow=True, ncol=5)

                        ButtonSaveNormalizedData = widgets.Button(
                            description="Save normalized data",
                            layout=Layout(width='30%', height='35px'))
                        display(ButtonSaveNormalizedData)

                        @ButtonSaveNormalizedData.on_click
                        def ActionSaveNormalizedData(selfbutton):
                            # Save normalized data
                            for j, C in enumerate(self.used_class_list):
                                temp_df = pd.DataFrame()
                                temp_df["Energy"] = energy[j][v1[j]:v2[j]]
                                temp_df["\u03BC"] = mu[j][v1[j]:v2[j]]
                                temp_df["background_corrected"] = ITB[j]
                                temp_df["\u03BC_variance"] = [
                                    1/d if d > 0 else 0 for d in ITB[j]]
                                temp_df["second_normalized_\u03BC"] = ITN[j]
                                setattr(C, "reduced_df", temp_df)
                                print(f"Saved Dataset {C.filename}")
                                temp_df.to_csv(
                                    f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

            # Polynoms
            controls = [widgets.IntSlider(
                description=f"P_{i+1}",
                min=v1[number],
                max=v2[number],
                step=1,
                orientation="vertical",
                continuous_update=False) for i in range(sL)]

            controls_dict = {c.description: c for c in controls}

            uif = widgets.HBox(tuple(controls))
            outf = widgets.interactive_output(plot_sliders, controls_dict)
            display(uif, outf)

        except (AttributeError, KeyError):
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")

        except (ValueError, NameError):
            plt.close()
            print("The selected energy range is wrong.")

    def reduce_single_spline(
        self,
        y,
        order,
        interval,
        cursor,
        param_A,
        param_B
    ):
        """Single spline method to remove background"""

        try:
            number = self.used_dataset_position
            df = self.used_df_name

            # Retrieve original data
            mu, energy, v1, v2, c = [], [], [], [], []
            for j, C in enumerate(self.used_class_list):
                used_df = getattr(C, df)
                mu.append(used_df[y].values)
                energy.append(used_df["Energy"].values)

                try:
                    v1.append(int(np.where(energy[j] == interval[0])[0]))
                except TypeError:
                    v1.append(0)

                try:
                    v2.append(int(np.where(energy[j] == interval[1])[0]))
                except TypeError:
                    v2.append(len(energy[j]) - 1)

                try:
                    c.append(int(np.where(energy[j] == cursor)[0]))
                except TypeError:
                    c.append(len(energy[j]) - 1)

            if order == "value":
                raise AttributeError("Please select an order.")

            elif order == "victoreen":
                # Make a lsq fit
                self.v_model = lmfit.Model(
                    self.victoreen, prefix='Background_')

                self.x = energy[number]
                self.y = mu[number]
                self.resultV = self.v_model.fit(mu[number][v1[number]:v2[number]], x=energy[number]
                                                [v1[number]:v2[number]], Background_A=int(param_A), Background_B=int(param_B))
                display(self.resultV.params)

                p = self.victoreen(
                    energy[number], self.resultV.params["Background_A"].value, self.resultV.params["Background_B"].value)

            elif isinstance(order, int):
                # Find the polynomials coefficients
                coef = np.polyfit(
                    energy[number][v1[number]:v2[number]], mu[number][v1[number]:v2[number]], order)

                # Create the polynomial function from the coefficients
                pcall = np.poly1d(coef)
                p = pcall(energy[number])

            # Substract background
            difference = mu[number]-p

            # Normalize
            normalized_data = difference / difference[c[number]]

            ButtonRemoveBackground = Button(
                description="Remove background for all",
                layout=Layout(width='30%', height='35px'))
            ButtonSaveDataset = Button(
                description="Save reduced data for this Dataset",
                layout=Layout(width='30%', height='35px'))
            display(widgets.HBox((ButtonRemoveBackground, ButtonSaveDataset)))

            plt.close()

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
            axs[0].set_xlabel("Energy")
            axs[0].set_ylabel('Intensity')
            axs[0].set_title('Raw Data')

            axs[0].plot(energy[number], mu[number], label='Data')
            axs[0].plot(energy[number][v1[number]:v2[number]], mu[number]
                        [v1[number]:v2[number]], '-o', label='Selected Region')
            axs[0].plot(energy[number], p, '--', color='green',
                        label='Background curve')
            axs[0].axvline(x=energy[number][v1[number]],
                           color='black', linestyle='--')
            axs[0].axvline(x=energy[number][v2[number]],
                           color='black', linestyle='--')
            axs[0].axvline(x=energy[number][c[number]], color='orange',
                           linestyle='--', label="cursor for normalization")

            axs[0].legend()

            axs[1].set_title('Background subtracted & normalized curve')
            axs[1].set_xlabel("Energy")
            axs[1].set_ylabel('Intensity')
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right()
            axs[1].set_xlim(energy[number][0], energy[number][-1])

            axs[1].plot(energy[number], normalized_data, '-', color='C0')

            @ButtonSaveDataset.on_click
            def ActionSaveDataset(selfbutton):
                # Save single Dataset without background in Class
                C = self.used_dataset
                temp_df = pd.DataFrame()
                temp_df["Energy"] = energy[number]
                temp_df["\u03BC"] = mu[number]
                temp_df["background_corrected"] = difference
                temp_df["\u03BC_variance"] = [
                    1 / d if d > 0 else 0 for d in difference]
                temp_df["second_normalized_\u03BC"] = normalized_data
                setattr(C, "reduced_df", temp_df)
                display(temp_df)
                print(f"Saved Dataset {C.filename}")
                temp_df.to_csv(
                    f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

            @ButtonRemoveBackground.on_click
            def ActionRemoveBackground(selfbutton):
                # Substract background to the intensity
                clear_output(True)

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                ax[0].set_title('Background subtracted')
                ax[0].set_xlabel("Energy")
                ax[0].set_ylabel('Intensity')

                ax[1].set_title('Background subtracted shifted')
                ax[1].set_xlabel("Energy")
                ax[1].set_ylabel('Intensity')
                ax[1].yaxis.tick_right()
                ax[1].yaxis.set_label_position("right")

                ITB = []
                try:
                    for i in range(len(mu)):
                        if order == "value":
                            raise AttributeError("Please select an order.")

                        elif order == "victoreen":
                            # Make a lsq fit
                            self.v_model = lmfit.Model(
                                self.victoreen, prefix='Background_')

                            self.x = energy[i]
                            self.y = mu[i]
                            self.resultV = self.v_model.fit(mu[i][v1[i]:v2[i]], x=energy[i][v1[i]:v2[i]], Background_A=int(
                                param_A), Background_B=int(param_B))
                            print(
                                f"Parameters for {self.used_class_list[i].filename}")
                            display(self.resultV.params)
                            print("\n")

                            p = self.victoreen(
                                energy[i], self.resultV.params["Background_A"].value, self.resultV.params["Background_B"].value)

                        elif isinstance(order, int):
                            # Find the polynomials coefficients
                            coef = np.polyfit(
                                energy[i][v1[i]:v2[i]], mu[i][v1[i]:v2[i]], order)

                            # Create the polynomial function from the coefficients
                            pcall = np.poly1d(coef)
                            p = pcall(energy[i])

                        ITB.append(mu[i] - p)

                    for i in range(len(ITB)):
                        ax[0].plot(energy[i], ITB[i])
                        ax[1].plot(energy[i], ITB[i]+0.1*(i),
                                   label=self.used_class_list[i].filename)

                    ax[1].legend(loc='upper center', bbox_to_anchor=(
                        0, -0.2), fancybox=True, shadow=True, ncol=5)

                except Exception as e:
                    print(e)

                ButtonSave = widgets.Button(
                    description="Save background reduced data",
                    layout=Layout(width='30%', height='35px'))
                ButtonNormalize = widgets.Button(
                    description='Normalize for all',
                    layout=Layout(width='30%', height='35px'))
                display(widgets.HBox((ButtonNormalize, ButtonSave)))

                @ButtonSave.on_click
                def ActionButtonSave(selfbutton):
                    # Save intensity without background
                    for j, C in enumerate(self.used_class_list):
                        temp_df = pd.DataFrame()
                        temp_df = getattr(C, "reduced_df")
                        temp_df["Energy"] = energy[j][v1[j]:v2[j]]
                        temp_df["\u03BC"] = mu[j][v1[j]:v2[j]]
                        temp_df["background_corrected"] = ITB[j]
                        temp_df["\u03BC_variance"] = [
                            1/d if d > 0 else 0 for d in ITB[j]]
                        setattr(C, "reduced_df", temp_df)
                        print(f"Saved Dataset {C.filename}")
                        temp_df.to_csv(
                            f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

                @ButtonNormalize.on_click
                def ActionButtonNormalize(selfbutton):
                    # Normalize data
                    clear_output(True)
                    area = []
                    ITN = []
                    for i in range(len(ITB)):
                        ITN.append(ITB[i] / ITB[i][c[i]])

                    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                    ax[0].set_title('Background subtracted normalized')
                    ax[0].set_xlabel("Energy")
                    ax[0].set_ylabel('Intensity')
                    ax[0].tick_params(direction='in', labelsize=15, width=2)
                    ax[1].set_title(
                        'Background subtracted normalized & shifted')
                    ax[1].set_xlabel("Energy")
                    ax[1].set_ylabel('Intensity')
                    ax[1].yaxis.set_label_position("right")
                    ax[1].yaxis.tick_right()
                    ax[1].tick_params(direction='in', labelsize=15, width=2)

                    for i in range(len(ITN)):
                        ax[0].plot(energy[i], ITN[i])
                        ax[1].plot(energy[i], ITN[i]+0.1*(i+1),
                                   label=self.used_class_list[i].filename)

                    ax[1].legend(loc='upper center', bbox_to_anchor=(
                        0, -0.2), fancybox=True, shadow=True, ncol=5)

                    ButtonSaveNormalizedData = widgets.Button(
                        description="Save normalized data",
                        layout=Layout(width='30%', height='35px'))
                    display(ButtonSaveNormalizedData)

                    @ButtonSaveNormalizedData.on_click
                    def ActionSaveNormalizedData(selfbutton):
                        # Save normalized data
                        for j, C in enumerate(self.used_class_list):
                            temp_df = pd.DataFrame()
                            temp_df["Energy"] = energy[j]
                            temp_df["\u03BC"] = mu[j]
                            temp_df["background_corrected"] = ITB[j]
                            temp_df["\u03BC_variance"] = [
                                1/d if d > 0 else 0 for d in ITB[j]]
                            temp_df["second_normalized_\u03BC"] = ITN[j]
                            setattr(C, "reduced_df", temp_df)
                            print(f"Saved Dataset {C.filename}")
                            temp_df.to_csv(
                                f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

        except (AttributeError, KeyError):
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")

        except (ValueError, NameError):
            plt.close()
            print("The selected energy range is wrong.")

    def reduce_splines_derivative(self, y, interval):
        """Finds the maximum of the derivative foe each Dataset"""

        def derivative_list(energy, mu):
            """
            Return the center point derivative for each point x_i as
            np.gradient(y) / np.gradient(x)
            """
            dEnergy, dIT = [], []

            for i in range(len(mu)):
                x = energy[i].values
                y = mu[i].values

                dEnergy.append(x)
                dIT.append(np.gradient(y) / np.gradient(x))

            return dEnergy, dIT

        try:
            number = self.used_dataset_position
            df = self.used_df_name

            mu, energy, v1, v2 = [], [], [], []
            for j, C in enumerate(self.used_class_list):
                used_df = getattr(C, df)
                mu.append(used_df[y])
                energy.append(used_df["Energy"])
                try:
                    v1.append(int(np.where(energy[j] == interval[0])[0]))
                except TypeError:
                    v1.append(0)

                try:
                    # Take one less point due to the right derivative in derivative_list
                    v2.append(int(np.where(energy[j] == interval[1])[0]) - 1)
                except TypeError:
                    v2.append(len(energy[j])-1 - 1)

            dE, dy = derivative_list(energy, mu)

            Emin = [np.min(e) for e in dE]
            Emax = [np.max(e) for e in dE]
            maxima = [k.argmax() for k in dy]

            def sliderCursor(s):
                plt.close()
                energymaximasl = dE[number][maxima[number]]

                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                axs[0].set_xlabel("Energy")
                axs[0].set_ylabel('Intensity')
                axs[0].set_title('1st Derivative')
                axs[0].tick_params(direction='in', labelsize=15, width=2)
                axs[0].plot(dE[number], dy[number], "--",
                            linewidth=1, label='Derivative')
                axs[0].set_xlim(Emin[number], Emax[number])

                maxD = np.max(dy[number])
                positionMaxD = list(dy[number]).index(maxD)

                axs[0].plot(dE[number][v1[number]:v2[number]], dy[number]
                            [v1[number]:v2[number]], linewidth=1, label='Selected Region')
                axs[0].plot(dE[number][positionMaxD], maxD, 'o',
                            markersize=2, label='E0 derivative')
                axs[0].axvline(x=dE[number][s], color='green',
                               linestyle='--', label='E0 slider')
                axs[0].axvline(x=dE[number][v1[number]],
                               color='black', linestyle='--')
                axs[0].axvline(x=dE[number][v2[number]],
                               color='black', linestyle='--')
                axs[0].legend()

                axs[1].plot(energy[number][v1[number]:v2[number]],
                            mu[number][v1[number]:v2[number]])
                axs[1].set_xlim(energy[number][v1[number]],
                                energy[number][v2[number]])
                axs[1].tick_params(direction='in', labelsize=15, width=2)
                axs[1].axvline(x=energy[number][s],
                               color='green', linestyle='--')
                axs[1].set_title("F")

                axs[1].set_xlabel("Energy")
                axs[1].set_ylabel('Intensity')
                axs[1].yaxis.set_label_position("right")

                print('cursor Position:', dE[number][s], 'eV')
                print("Calculated Maximum Position:",
                      dE[number][positionMaxD], 'eV', ' ; channel:', positionMaxD)

                ButtonSaveE0 = widgets.Button(
                    description="Save E0",
                    layout=Layout(width='25%', height='35px'))
                ButtonSaveAll = widgets.Button(
                    description="Save all default values",
                    layout=Layout(width='25%', height='35px'))
                ButtonSplinesReduction = widgets.Button(
                    description="Proceed to reduction",
                    layout=Layout(width='25%', height='35px'))
                display(widgets.HBox(
                    (ButtonSaveE0, ButtonSaveAll, ButtonSplinesReduction)))

                def ActionSaveE0(selfbutton):
                    setattr(self.used_dataset, "E0", dE[number][s])
                    print(f"Saved E0 for {self.used_dataset.filename};  ")

                def ActionSaveAll(selfbutton):
                    for j, C in enumerate(self.used_class_list):
                        setattr(
                            self.used_class_list[j], "E0", dE[j][maxima[j]])
                        print(
                            f"Saved E0 for {self.used_class_list[j].filename};  ")

                def ActionSplinesReduction(selfbutton):
                    try:
                        E0Values = [getattr(self.used_class_list[j], "E0")
                                    for j, C in enumerate(self.used_class_list)]

                        self._list_reduce_splines = interactive(
                            self.reduce_splines,
                            used_dataset=widgets.Dropdown(
                                options=self.used_class_list,
                                description='Select the Dataset:',
                                disabled=False,
                                style={'description_width': 'initial'},
                                layout=Layout(width='60%')),
                            order_pre=widgets.Dropdown(
                                options=[
                                    ("Select and order", "value"),
                                    ("Victoreen", "victoreen"),
                                    ("0", 0),
                                    ("1", 1),
                                    ("2", 2),
                                    ("3", 3)
                                ],
                                value="value",
                                description='Order of pre-edge:',
                                disabled=False,
                                style={'description_width': 'initial'}),
                            order_pst=widgets.Dropdown(
                                options=[
                                    ("Select and order", "value"),
                                    ("Victoreen", "victoreen"),
                                    ("0", 0),
                                    ("1", 1),
                                    ("2", 2),
                                    ("3", 3)
                                ],
                                value="value",
                                description='Order of post-edge:',
                                disabled=False,
                                style={'description_width': 'initial'}),
                            s1=widgets.FloatRangeSlider(
                                min=self.new_energy_column[0],
                                value=[self.new_energy_column[0], np.round(
                                    self.new_energy_column[0] + 0.33*(self.new_energy_column[-1] - self.new_energy_column[0]), 0)],
                                max=self.new_energy_column[-1],
                                step=self.interpol_step,
                                description='Energy range (eV):',
                                disabled=False,
                                continuous_update=False,
                                orientation="horizontal",
                                readout=True,
                                readout_format='.2f',
                                style={'description_width': 'initial'},
                                layout=Layout(width="50%", height='40px')),
                            s2=widgets.FloatRangeSlider(
                                min=self.new_energy_column[0],
                                value=[np.round(self.new_energy_column[0] + 0.66*(
                                    self.new_energy_column[-1] - self.new_energy_column[0]), 0), self.new_energy_column[-1]],
                                max=self.new_energy_column[-1],
                                step=self.interpol_step,
                                description='Energy range (eV):',
                                disabled=False,
                                continuous_update=False,
                                orientation="horizontal",
                                readout=True,
                                readout_format='.2f',
                                style={'description_width': 'initial'},
                                layout=Layout(width="50%", height='40px')),
                            param_a1=widgets.Text(
                                value="1000000000",
                                placeholder='A1 = ',
                                description='A1:',
                                disabled=True,
                                continuous_update=False,
                                style={'description_width': 'initial'}),
                            param_b1=widgets.Text(
                                value="1000000000",
                                placeholder='B1 = ',
                                description='B1:',
                                disabled=True,
                                continuous_update=False,
                                style={'description_width': 'initial'}),
                            param_a2=widgets.Text(
                                value="1000000000",
                                placeholder='A2 = ',
                                description='A2:',
                                disabled=True,
                                continuous_update=False,
                                style={'description_width': 'initial'}),
                            param_b2=widgets.Text(
                                value="1000000000",
                                placeholder='B2 = ',
                                description='B2:',
                                disabled=True,
                                continuous_update=False,
                                style={'description_width': 'initial'}),
                            y=fixed(y))
                        self.widget_list_reduce_splines = widgets.VBox([
                            self._list_reduce_splines.children[0], widgets.HBox(
                                self._list_reduce_splines.children[1:3]),
                            widgets.HBox(self._list_reduce_splines.children[3:5]), widgets.HBox(
                                self._list_reduce_splines.children[5:9]),
                            self._list_reduce_splines.children[-1]
                        ])

                        self._list_tab_reduce_method.children[1].disabled = True
                        self._list_reduce_splines_derivative.children[0].disabled = True
                        self._list_reduce_splines_derivative.children[1].disabled = True
                        self.sliders_splines.children[0].disabled = True

                        self._list_reduce_splines.children[1].observe(
                            self.param_victoreen_handler_1, names="value")
                        self._list_reduce_splines.children[2].observe(
                            self.param_victoreen_handler_2, names="value")

                        clear_output(True)
                        print(
                            f"We now use the values previously fixed for E0 in our reduction routine, to normalize the intensity by the absorption edge jump.\n")
                        display(self.widget_list_reduce_splines)
                    except Exception as e:
                        raise e
                    # except AttributeError:
                    #     print("You have not yet fixed all the values !")

                ButtonSplinesReduction.on_click(ActionSplinesReduction)
                ButtonSaveE0.on_click(ActionSaveE0)
                ButtonSaveAll.on_click(ActionSaveAll)

            self.sliders_splines = interactive(sliderCursor,
                                               s=widgets.BoundedIntText(
                                                   value=maxima[number],
                                                   step=1,
                                                   min=0,
                                                   max=len(energy[0]) - 1,
                                                   description='cursor:',
                                                   disabled=False))
            display(self.sliders_splines)

        except (AttributeError, KeyError):
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")
        except (ValueError, NameError):
            plt.close()
            print("The selected energy range is wrong.")

    def reduce_splines(
        self,
        used_dataset,
        order_pre,
        order_pst,
        s1,
        s2,
        param_a1,
        param_b1,
        param_a2,
        param_b2,
        y
    ):
        """
        Reduce the background using two curves and then normalize by edge-jump.
        """

        try:
            number = self.used_class_list.index(used_dataset)
            df = self.used_df_name
            print(s1, s2)

            mu, energy, E0, v1, v2, v3, v4 = [], [], [], [], [], [], []
            for j, C in enumerate(self.used_class_list):
                used_df = getattr(C, df)
                mu.append(used_df[y].values)
                energy.append(used_df["Energy"].values)
                E0.append(getattr(C, "E0"))

                # First and second zoom
                try:
                    v1.append(int(np.where(energy[j] == s1[0])[0]))
                except TypeError:
                    v1.append(0)

                try:
                    # Take one less point due to the right derivative in derivative_list
                    v2.append(int(np.where(energy[j] == s1[1])[0]) - 1)
                except TypeError:
                    v2.append(len(energy[j])-1 - 1)

                try:
                    v3.append(int(np.where(energy[j] == s2[0])[0]))
                except TypeError:
                    v3.append(0)

                try:
                    # Take one less point due to the right derivative in derivative_list
                    v4.append(int(np.where(energy[j] == s2[1])[0]) - 1)
                except TypeError:
                    v4.append(len(energy[j])-1 - 1)

            plt.close()

            # Compute the background that will be subtracted
            e0 = min(energy[number], key=lambda x: abs(x-E0[number]))
            e0c = list(energy[number]).index(e0)

            if order_pre == "value":
                raise AttributeError("Please select an order.")

            elif order_pre == "victoreen":
                # Make a lsq fit
                self.v_model = lmfit.Model(self.victoreen, prefix='Pre_edge_')

                self.x = energy[number]
                self.y = mu[number]
                self.result_victoreen_pre_edge = self.v_model.fit(
                    mu[number][v1[number]:v2[number]], x=energy[number][v1[number]:v2[number]], Pre_edge_A=int(param_a1), Pre_edge_B=int(param_b1))
                display(self.result_victoreen_pre_edge.params)

                p1 = self.victoreen(energy[number], self.result_victoreen_pre_edge.params["Pre_edge_A"].value,
                                    self.result_victoreen_pre_edge.params["Pre_edge_B"].value)
                p1E0 = self.victoreen(E0[number], self.result_victoreen_pre_edge.params["Pre_edge_A"].value,
                                      self.result_victoreen_pre_edge.params["Pre_edge_B"].value)

            elif isinstance(order_pre, int):
                # Find the polynomials coefficients
                coef1 = np.polyfit(
                    energy[number][v1[number]:v2[number]], mu[number][v1[number]:v2[number]], order_pre)

                # Create the polynomial function from the coefficients
                p1call = np.poly1d(coef1)
                p1 = p1call(energy[number])
                p1E0 = p1call(E0[number])

            if order_pst == "value":
                raise AttributeError("Please select an order.")

            elif order_pst == "victoreen":
                # Make a lsq fit
                self.v_model = lmfit.Model(self.victoreen, prefix='Post_edge_')

                self.result_victoreen_post_edge = self.v_model.fit(
                    mu[number][v3[number]:v4[number]], x=energy[number][v3[number]:v4[number]], Post_edge_A=int(param_a2), Post_edge_B=int(param_b2))
                display(self.result_victoreen_post_edge.params)

                p2 = self.victoreen(energy[number], self.result_victoreen_post_edge.params["Post_edge_A"].value,
                                    self.result_victoreen_post_edge.params["Post_edge_B"].value)
                p2E0 = self.victoreen(E0[number], self.result_victoreen_post_edge.params["Post_edge_A"].value,
                                      self.result_victoreen_post_edge.params["Post_edge_B"].value)

            elif isinstance(order_pst, int):
                # Find the polynomials coefficients
                coef2 = np.polyfit(
                    energy[number][v3[number]:v4[number]], mu[number][v3[number]:v4[number]], order_pst)

                # Create the polynomial function from the coefficients
                p2call = np.poly1d(coef2)
                p2 = p2call(energy[number])
                p2E0 = p2call(E0[number])

            # Substract pre-edge
            ITs = mu[number] - p1
            # Compute edge-jump
            delta = abs(p2E0 - p1E0)
            # Normalise
            ITB = ITs / delta
            ITN = ITB / np.trapz(ITB)

            # Plot current work
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
            axs[0].set_xlabel("Energy")
            axs[0].set_ylabel('Intensity')
            axs[0].set_title('Raw Data')
            axs[0].tick_params(direction='in', labelsize=15, width=2)

            axs[0].plot(energy[number], mu[number], linewidth=1, label='Data')
            axs[0].plot(energy[number][v1[number]:v2[number]], mu[number]
                        [v1[number]:v2[number]], 'o', color='orange', label='pre-edge')
            axs[0].plot(energy[number][v3[number]:v4[number]], mu[number]
                        [v3[number]:v4[number]], 'o', color='red', label='post-edge')
            axs[0].axvline(energy[number][e0c],  color='green',
                           linestyle='--', label='E0')

            axs[0].axvline(energy[number][v1[number]],
                           color='orange', linestyle='--')
            axs[0].axvline(energy[number][v2[number]],
                           color='orange', linestyle='--')
            axs[0].axvline(energy[number][v3[number]],
                           color='tomato', linestyle='--')
            axs[0].axvline(energy[number][v4[number]],
                           color='tomato', linestyle='--')

            axs[0].plot(energy[number], p1, '--', linewidth=1,
                        color='dodgerblue', label='Polynoms')
            axs[0].plot(energy[number], p2, '--',
                        linewidth=1, color='dodgerblue')

            axs[0].legend()
            axs[0].set_ylim(np.min(mu[number])-0.01*np.min(mu[number]),
                            np.max(mu[number])+0.01*np.max(mu[number]))

            # Plot without background
            axs[1].set_title('Bkg subtracted & normalized')

            axs[1].plot(energy[number], ITB, label='Data')

            axs[1].axvline(E0[number], color='green',
                           linestyle='--', label="E0")
            axs[1].axhline(1, color='red', linestyle='--',
                           label="Normalization to 1.")
            axs[1].set_xlim(np.min(energy[number]), np.max(energy[number]))
            axs[1].set_xlabel("Energy")
            axs[1].set_ylabel('Intensity')
            axs[1].yaxis.set_label_position("right")
            axs[1].tick_params(direction='in', labelsize=15, width=2)
            axs[1].legend()

            ButtonSaveDataset = Button(
                description="Save reduced data for this Dataset",
                layout=Layout(width='50%', height='35px'))
            display(ButtonSaveDataset)

            @ButtonSaveDataset.on_click
            def ActionSaveDataset(selfbutton):
                # Save single Dataset without background in Class
                temp_df = pd.DataFrame()
                temp_df["Energy"] = energy[number]
                temp_df["\u03BC"] = mu[number]
                temp_df["background_corrected"] = ITB
                temp_df["\u03BC_variance"] = [1/d if d > 0 else 0 for d in ITB]
                temp_df["second_normalized_\u03BC"] = ITN
                display(temp_df)
                setattr(used_dataset, "reduced_df_splines", temp_df)
                print(f"Saved Dataset {used_dataset.filename}")
                temp_df.to_csv(
                    f"{self.folders[2]}{used_dataset.filename}_SplinesReduced.csv", index=False)

                # Need to plot again
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                axs[0].set_xlabel("Energy")
                axs[0].set_ylabel('Intensity')
                axs[0].set_title('Raw Data')
                axs[0].tick_params(direction='in', labelsize=15, width=2)

                axs[0].plot(energy[number], mu[number],
                            linewidth=1, label='Data')
                axs[0].plot(energy[number][v1[number]:v2[number]], mu[number]
                            [v1[number]:v2[number]], 'o', color='orange', label='pre-edge')
                axs[0].plot(energy[number][v3[number]:v4[number]], mu[number]
                            [v3[number]:v4[number]], 'o', color='red', label='post-edge')
                axs[0].axvline(energy[number][e0c],  color='green',
                               linestyle='--', label='E0')

                axs[0].axvline(energy[number][v1[number]],
                               color='orange', linestyle='--')
                axs[0].axvline(energy[number][v2[number]],
                               color='orange', linestyle='--')
                axs[0].axvline(energy[number][v3[number]],
                               color='tomato', linestyle='--')
                axs[0].axvline(energy[number][v4[number]],
                               color='tomato', linestyle='--')

                axs[0].plot(energy[number], p1, '--', linewidth=1,
                            color='dodgerblue', label='Polynoms')
                axs[0].plot(energy[number], p2, '--',
                            linewidth=1, color='dodgerblue')

                axs[0].legend()
                axs[0].set_ylim(np.min(mu[number])-0.01*np.min(mu[number]),
                                np.max(mu[number])+0.01*np.max(mu[number]))

                # Plot without background
                axs[1].set_title('Bkg subtracted & normalized')

                axs[1].plot(energy[number], ITB, label='Data')

                axs[1].axvline(E0[number], color='green',
                               linestyle='--', label="E0")
                axs[1].axhline(1, color='red', linestyle='--',
                               label="Normalization to 1.")
                axs[1].set_xlim(np.min(energy[number]), np.max(energy[number]))
                axs[1].set_xlabel("Energy")
                axs[1].set_ylabel('Intensity')
                axs[1].yaxis.set_label_position("right")
                axs[1].tick_params(direction='in', labelsize=15, width=2)
                axs[1].legend()
                plt.tight_layout()

                plt.savefig(
                    f"{self.folders[3]}splines_reduced_{used_dataset.filename}.pdf")
                plt.savefig(
                    f"{self.folders[3]}splines_reduced_{used_dataset.filename}.png")
                plt.close()

        except (AttributeError, KeyError):
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")
        except (ValueError, NameError):
            plt.close()
            print("The selected energy range, order, or parameter value is wrong.")

    def normalize_maxima(self, y, interval):
        """
        """

        try:
            number = self.used_dataset_position
            df = self.used_df_name

            # Retrieve original data
            mu, energy, v1, v2 = [], [], [], []
            for j, C in enumerate(self.used_class_list):
                used_df = getattr(C, df)
                mu.append(used_df[y].values)
                energy.append(used_df["Energy"].values)

                try:
                    v1.append(int(np.where(energy[j] == interval[0])[0]))
                except TypeError:
                    v1.append(0)

                try:
                    v2.append(int(np.where(energy[j] == interval[1])[0]))
                except TypeError:
                    v2.append(len(energy[j])-1)

            plt.close()

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
            axs[0].set_xlabel("Energy")
            axs[0].set_ylabel('Intensity')
            axs[0].set_title('Data')
            axs[0].tick_params(direction='in', labelsize=15, width=2)

            axs[0].plot(energy[number], mu[number], label='Data')
            axs[0].plot(energy[number][v1[number]:v2[number]], mu[number]
                        [v1[number]:v2[number]], '-o', label='Selected Region')
            axs[0].axvline(x=energy[number][v1[number]],
                           color='black', linestyle='--')
            axs[0].axvline(x=energy[number][v2[number]],
                           color='black', linestyle='--')
            axs[0].legend()

            axs[1].set_title('normalized data')
            axs[1].set_xlabel("Energy")
            axs[1].set_ylabel('Intensity')
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right()
            axs[1].tick_params(direction='in', labelsize=15, width=2)
            axs[1].set_xlim(energy[number][v1[number]],
                            energy[number][v2[number]])

            axs[1].plot(
                energy[number][v1[number]:v2[number]],
                mu[number][v1[number]:v2[number]] /
                max(mu[number][v1[number]:v2[number]]),
                '-',
                color='C0'
            )

            print("Channel 1:", v1[number], ";",
                  "energy:", energy[number][v1[number]])
            print("Channel 2:", v2[number], ";",
                  "energy:", energy[number][v2[number]])

            ButtonNormMax = Button(
                description="Normalize all spectra by their maximum intensity.",
                layout=Layout(width='30%', height='35px'))
            display(ButtonNormMax)

            @ButtonNormMax.on_click
            def ActionNormMax(selfbutton):

                plt.close()

                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                axs[0].set_xlabel("Energy")
                axs[0].set_ylabel('Intensity')
                axs[0].set_title('Data')

                axs[1].set_title('normalized data')
                axs[1].set_xlabel("Energy")
                axs[1].set_ylabel('Intensity')

                normalized_data = []
                for j, C in enumerate(self.used_class_list):
                    axs[0].plot(energy[j][v1[j]:v2[j]], mu[j]
                                [v1[j]:v2[j]], label=f"{C.filename}")
                    axs[1].plot(energy[j][v1[j]:v2[j]], (mu[j][v1[j]:v2[j]] /
                                max(mu[j][v1[j]:v2[j]])), label=f"{C.filename}")
                    normalized_data.append(
                        mu[j][v1[j]:v2[j]] / max(mu[j][v1[j]:v2[j]]))

                axs[1].legend(loc='upper center', bbox_to_anchor=(
                    0, -0.2), fancybox=True, shadow=True, ncol=1)
                plt.show()

                ButtonSaveNormalizedData = widgets.Button(
                    description="Save normalized data",
                    layout=Layout(width='30%', height='35px'))
                display(ButtonSaveNormalizedData)

                @ButtonSaveNormalizedData.on_click
                def ActionSaveNormalizedData(selfbutton):
                    # Save normalized data
                    for j, C in enumerate(self.used_class_list):
                        temp_df = pd.DataFrame()
                        temp_df["Energy"] = energy[j][v1[j]:v2[j]]
                        temp_df["\u03BC"] = mu[j][v1[j]:v2[j]]
                        temp_df["\u03BC_variance"] = [1/d if d >
                                                      0 else 0 for d in mu[j][v1[j]:v2[j]]]
                        temp_df["second_normalized_\u03BC"] = normalized_data[j]
                        setattr(C, "reduced_df", temp_df)
                        print(f"Saved Dataset {C.filename}")
                        temp_df.to_csv(
                            f"{self.folders[2]}{C.filename}_reduced.csv", index=False)

        except (AttributeError, KeyError):
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")

        except (ValueError, NameError):
            plt.close()
            print("The selected energy range is wrong.")

    # Fitting
    def define_fitting_df(self, used_dataset, df, show_bool):
        """
        Make sure that the DataFrame we want to use exist, and initiate the
        fitting routine.

        :param used_dataset:
        :param df:
        :param show_bool:
        """

        if not show_bool:
            print("Window cleared")
            clear_output(True)

        elif show_bool:
            try:
                self.used_dataset = used_dataset
                self.used_df_name = df

                # test here is df is accessible
                self.used_df = getattr(
                    self.used_dataset, self.used_df_name)

                display(self.widget_list_define_model)

            except (AttributeError, KeyError):
                print(f"Wrong Dataset and column combination !")

    def define_model(
        self,
        x_axis,
        y_axis,
        interval,
        peak_number,
        peak_type,
        background_type,
        pol_degree,
        step_type,
        method,
        w,
        fix_model
    ):
        """
        We built a model using the lmfit package, composed of a background,
        a step and a certain number of polynomials

        :param x_axis:
        :param y_axis:
        :param interval:
        :param peak_number:
        :param peak_type:
        :param background_type:
        :param pol_degree:
        :param step_type:
        :param method:
        :param w: weights used in the fitting routine
        :param fix_model:
        """
        clear_output(True)

        try:
            # Get data on interval
            v1, v2 = interval
            indices = np.where(
                (self.used_df[x_axis] > v1) &
                (self.used_df[x_axis] < v2))

            y = self.used_df[y_axis].values[indices[0]]
            x = self.used_df[x_axis].values[indices[0]]

            # Initialize model and parameters, add Background
            if background_type == PolynomialModel:
                self.mod = background_type(degree=pol_degree, prefix='Bcgd_')
                self.pars = self.mod.guess(y, x=x)

            elif background_type == "victoreen":
                self.mod = lmfit.Model(self.victoreen, prefix='Bcgd_')
                self.pars = self.mod.make_params(Bcgd_A=1, Bcgd_B=1)

            else:
                self.mod = background_type(prefix='Bcgd_')
                self.pars = self.mod.guess(y, x=x)

            # Add a step if needed
            if step_type:
                Step = StepModel(form=step_type, prefix="Step_")
                self.pars.update(Step.make_params())
                self.mod += Step

            # Create a dictionnary for the peak to iterate on their names
            peaks = dict()

            for i in range(peak_number):
                peaks[f"Peak_{i}"] = peak_type(prefix=f"P{i}_")
                self.pars.update(peaks[f"Peak_{i}"].make_params())
                self.mod += peaks[f"Peak_{i}"]

            if fix_model:
                def InitPara(para, column, value):
                    """
                    """
                    ButtonRetrievePara = Button(
                        description="Retrieve parameters",
                        layout=Layout(width='25%', height='35px'))
                    ButtonSavePara = Button(
                        description="Save parameter value",
                        layout=Layout(width='25%', height='35px'))
                    ButtonFit = Button(
                        description="Launch Fit",
                        layout=Layout(width='15%', height='35px'))
                    ButtonGuess = Button(
                        description="See current guess",
                        layout=Layout(width='15%', height='35px'))
                    ButtonSaveModel = Button(
                        description="Save current work",
                        layout=Layout(width='15%', height='35px'))
                    display(widgets.HBox((
                        ButtonRetrievePara, ButtonSavePara, ButtonGuess,
                        ButtonFit, ButtonSaveModel
                    )))
                    display(self.pars)

                    @ButtonRetrievePara.on_click
                    def ActionRetrievePara(selfbutton):
                        clear_output(True)

                        # Load parameters from previous fitting result
                        try:
                            self.pars = getattr(
                                self.used_dataset,
                                self.used_df_name[:-3] + "_fit_result").params.copy()
                            print("Previously saved parameters loaded.")

                            ButtonGuess.click()
                        except:
                            print("Could not load any parameters.")

                    @ButtonSavePara.on_click
                    def ActionSavePara(selfbutton):
                        clear_output(True)
                        try:
                            if column == "value":
                                self.pars[f"{para}"].set(value=value)
                            if column == "min":
                                self.pars[f"{para}"].set(min=value)
                            if column == "max":
                                self.pars[f"{para}"].set(max=value)

                            ButtonGuess.click()
                        except Exception as e:
                            raise e

                    @ButtonGuess.on_click
                    def ActionGuess(selfbutton):
                        clear_output(True)
                        display(widgets.HBox((
                            ButtonRetrievePara, ButtonSavePara,
                            ButtonGuess, ButtonFit, ButtonSaveModel)))
                        try:
                            display(self.pars)

                            # Current guess
                            self.init = self.mod.eval(self.pars, x=x)

                            # Bokeh plot
                            source = ColumnDataSource(
                                data=dict(
                                    x=x,
                                    y=y,
                                    y_guess=self.init,
                                ))

                            source_guess = ColumnDataSource(
                                data=dict(
                                    x=x,
                                    y_guess=self.init,
                                ))

                            # Create figure
                            TOOLTIPS = [
                                (f"{x_axis} (eV)", "$x"),
                                (y_axis, "$y"),
                            ]

                            p = figure(
                                height=600, width=self.figure_width,
                                tools="xpan, pan, wheel_zoom, box_zoom, reset, undo, redo, crosshair, hover, save",
                                tooltips=TOOLTIPS,
                                active_scroll="wheel_zoom",
                                x_axis_label=x_axis + "(eV)",
                                y_axis_label=y_axis
                            )

                            p.line("x", "y", source=source, line_alpha=0.8,
                                   legend_label="Data",
                                   line_color=self.matplotlib_colours[0],
                                   hover_line_alpha=1.0, hover_line_width=2.0,
                                   muted_alpha=0.1)
                            p.line("x", "y_guess", source=source_guess,
                                   legend_label="Initial guess", line_alpha=0.8,
                                   line_color=self.matplotlib_colours[1],
                                   hover_line_alpha=1.0, hover_line_width=2.0,
                                   muted_alpha=0.1)

                            # Evaluate components
                            self.components = self.mod.eval_components(x=x)

                            if background_type == ConstantModel:
                                p.line(
                                    x=x,
                                    y=np.ones(len(x)) *
                                    self.components['Bcgd_'],
                                    line_alpha=0.5,
                                    line_color=self.matplotlib_colours[3],
                                    hover_line_alpha=1.0, hover_line_width=1.5,
                                    muted_alpha=0.1,
                                    legend_label='Background')
                            else:
                                p.line(
                                    x=x,
                                    y=self.components['Bcgd_'],
                                    line_alpha=0.5,
                                    line_color=self.matplotlib_colours[3],
                                    hover_line_alpha=1.0, hover_line_width=1.5,
                                    muted_alpha=0.1,
                                    legend_label='Background')

                            if step_type:
                                p.line(
                                    x=x,
                                    y=self.components['Step_'],
                                    line_alpha=0.5,
                                    line_color=self.matplotlib_colours[3],
                                    hover_line_alpha=1.0, hover_line_width=1.5,
                                    muted_alpha=0.1,
                                    legend_label='Step')

                            for i in range(peak_number):
                                p.line(
                                    x=x,
                                    y=self.components[f"P{i}_"],
                                    line_alpha=0.5,
                                    line_color=self.matplotlib_colours[3+i],
                                    hover_line_alpha=1.0, hover_line_width=1.5,
                                    muted_alpha=0.1,
                                    legend_label=f"Peak nb {i}")

                            # Hide plot by clicking on legend
                            p.legend.click_policy = "mute"

                            # Show figure
                            show(p)

                        except Exception as e:
                            raise e

                    @ButtonFit.on_click
                    def ActionFit(selfbutton):
                        """
                        See docs:
                           https://docs.scipy.org/doc/scipy/reference/
                           generated/scipy.stats.chisquare.html
                        """
                        clear_output(True)
                        display(widgets.HBox((
                            ButtonRetrievePara, ButtonSavePara,
                            ButtonGuess, ButtonFit, ButtonSaveModel)))

                        # Current guess
                        self.init = self.mod.eval(self.pars, x=x)

                        # Retrieve weights
                        # Possible to explore other options here
                        weights = 1/y.values if w == "Obs" else None

                        # Launch fit
                        self.out = self.mod.fit(
                            y,
                            self.pars,
                            x=x,
                            method=method,
                            weights=weights,
                        )
                        display(self.out.result)

                        # Compute the contribution of each component in the
                        # model for plots
                        self.components = self.out.eval_components(x=x)

                        # Plot
                        fig, axes = plt.subplots(
                            2, 2, figsize=(12, 7),
                            gridspec_kw={'height_ratios': [5, 1]})

                        axes[0, 0].plot(x, y, label="Data")
                        axes[0, 0].plot(x, self.out.best_fit, label='Best fit')
                        axes[0, 0].set_xlabel(x_axis, fontweight='bold')
                        axes[0, 0].set_ylabel(y_axis, fontweight='bold')
                        axes[0, 0].set_title(
                            f"Best fit - {self.used_dataset.filename}")
                        axes[0, 0].legend()

                        # Residuals
                        axes[1, 0].set_title("Residuals")
                        axes[1, 0].scatter(x, self.out.residual, s=0.5)
                        axes[1, 0].set_xlabel(x_axis, fontweight='bold')
                        axes[1, 0].set_ylabel(y_axis, fontweight='bold')

                        axes[1, 1].set_title("Residuals")
                        axes[1, 1].scatter(x, self.out.residual, s=0.5)
                        axes[1, 1].set_xlabel(x_axis, fontweight='bold')
                        axes[1, 1].set_ylabel(y_axis, fontweight='bold')

                        # Detailed plot
                        axes[0, 1].set_title("Best fit - Detailed")
                        axes[0, 1].plot(x, y, label="Data")

                        if background_type == ConstantModel:
                            axes[0, 1].plot(
                                x,
                                np.ones(len(x)) * self.components['Bcgd_'],
                                'k--',
                                label='Background')
                        else:
                            axes[0, 1].plot(
                                x,
                                self.components['Bcgd_'],
                                'k--',
                                label='Background')

                        if step_type:
                            axes[0, 1].plot(
                                x, self.components['Step_'], label='Step')

                        for i in range(peak_number):
                            axes[0, 1].plot(
                                x, self.components[f"P{i}_"], label=f"Peak nb {i}")

                        axes[0, 1].set_xlabel(x_axis, fontweight='bold')
                        axes[0, 1].set_ylabel(y_axis, fontweight='bold')
                        axes[0, 1].legend()

                        # Save figure
                        plt.tight_layout()
                        plt.savefig(f"{self.folders[3]}fit.pdf")
                        plt.savefig(f"{self.folders[3]}fit.png")
                        print(f"Saved image as {self.folders[3]}fit.pdf")

                        # Check the stats of the fit
                        try:
                            chisq, p = chisquare(
                                self.out.data,
                                self.out.best_fit,
                                ddof=(self.out.nfree)
                            )
                            setattr(self.used_dataset, "chisq", chisq)
                            setattr(self.used_dataset, "p", p)

                            print(
                                "#############################################"
                                "\nSum of squared residuals: "
                                f"{np.sum(self.out.residual**2):.3f}"
                                f", lmfit chisqr: {self.out.chisqr:.3f}"
                                f"\nSum of squared residuals/nfree: "
                                f"{np.sum(self.out.residual**2)/(self.out.nfree):.3f}"
                                f", lmfit redchisqr: {self.out.redchi:.3f}"
                                f"\nScipy Chi square for Poisson distri: {chisq:.3f}"
                                f", 1 - p = {1 - p:.3f}"
                                "\nlmfit chisqr divided iter by expected "
                                f"{np.sum((self.out.residual**2)/self.out.best_fit):.3f}\n"
                                "#############################################"
                            )
                        except ValueError:
                            print("Could not compute chi square (scipy.chisquare)")

                        # R factor
                        r_factor = 100 * \
                            (np.sum(self.out.residual**2)/np.sum(self.out.data**2))
                        self.used_dataset.r_factor = r_factor

                        print(
                            "\n#############################################\n"
                            f"R factor : {r_factor} %.\n"
                            "#############################################"
                        )

                        ButtonCI = Button(
                            description="Determine confidence Intervals",
                            layout=Layout(width='25%', height='35px'))
                        ButtonParaSpace = Button(
                            description="Determine the parameter distribution",
                            layout=Layout(width='35%', height='35px'))
                        display(widgets.HBox((ButtonCI,  ButtonParaSpace)))

                        @ButtonCI.on_click
                        def ActionCI(selfbutton):
                            """
                            The F-test is used to compare our null model, which
                            is the best fit we have found, with an alternate
                            model, where one of the parameters is fixed to a
                            specific value. For most models, it is not necessary
                            since the estimation of the standard error from the
                            estimated covariance matrix is normally quite good.
                            But for some models, the sum of two exponentials for
                            example, the approximation begins to fail.
                            Then use this method.
                            """
                            try:
                                # Confidence interval with the standard error
                                # from the covariance matrix
                                print(
                                    "The shape of the estimated covariance "
                                    f"matrix is : {np.shape(self.out.covar)}.\n"
                                )
                                print(self.out.ci_report())

                            except:
                                print(
                                    "No covariance matrix could be estimated "
                                    "from the fitting routine.\n"
                                    "We determined the confidence intervals "
                                    "without standard error estimates.\n"
                                    "Please refer to lmfit documentation for "
                                    "additional informations.\n"
                                    "We set the standard error to 10 % of the "
                                    "parameter values."
                                )

                                # Determine confidence intervals without
                                # standard error estimates, careful !
                                for p in self.out.result.params:
                                    self.out.result.params[p].stderr = abs(
                                        self.out.result.params[p].value * 0.1)
                                ci = lmfit.conf_interval(
                                    self.out, self.out.result)
                                lmfit.printfuncs.report_ci(ci)

                        @ButtonParaSpace.on_click
                        def ActionParaSpace(selfbutton):
                            return self.explore_params(i, j, x_axis, y_axis)

                    @ButtonSaveModel.on_click
                    def ActionSave(selfbutton):
                        clear_output(True)
                        display(widgets.HBox((
                            ButtonRetrievePara, ButtonSavePara, ButtonGuess,
                            ButtonFit, ButtonSaveModel)))

                        try:
                            # Save best parameter values as initial guess
                            self.pars = self.out.params

                            # Save fit output
                            setattr(
                                self.used_dataset,
                                self.used_df_name[:-3] + "_fit_result",
                                self.out.result)
                            print("Saved the output of the fitting routine.")

                            # Save result as dataframe
                            fit_df = pd.DataFrame({
                                x_axis: self.out.userkws["x"],
                                y_axis: self.out.data,
                                "fit": self.out.best_fit,
                                "residuals": self.out.residual,
                            })

                            # Save weights

                            # Save DF
                            setattr(
                                self.used_dataset,
                                self.used_df_name[:-3] + "_fit_df",
                                fit_df)

                            # Pickle dataset
                            self.used_dataset.pickle_dataset(
                                self.folders[1] +
                                self.used_dataset.filename.split(".")[
                                    0]+".pickle"
                            )

                        except AttributeError:
                            print("Launch the fit first. \n")
                        except Exception as e:
                            raise e

                    ButtonGuess.click()

                self._list_parameters_fit = interactive(
                    InitPara,
                    para=widgets.Dropdown(
                        options=[str(p) for p in self.pars],
                        value=None,
                        description='Select the parameter:',
                        style={'description_width': 'initial'}),
                    column=widgets.Dropdown(
                        options=["value", "min", "max"],
                        description='Select the column:',
                        style={'description_width': 'initial'}),
                    value=widgets.FloatText(
                        value=0,
                        step=0.01,
                        description='Value :'))
                self.widget_list_parameters_fit = widgets.VBox([
                    widgets.HBox(self._list_parameters_fit.children[0:3]),
                    self._list_parameters_fit.children[-1]
                ])
                display(self.widget_list_parameters_fit)

            else:
                plt.close()
                print("Cleared")
                clear_output(True)

        except (AttributeError, KeyError):
            plt.close()
            if y_axis == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")

    def explore_params(self, i, j, x_axis, y_axis):
        """
        To execute after a fit, allows one to explore the parameter space
        with the emcee Markov Monte carlo chain.

        This method does not actually perform a fit. Instead, it explores the
        parameter space to determine the probability distributions for the
        parameters, but without an explicit goal of attempting to refine the
        solution.

        To use this method effectively, you should first use another
        minimization method and then use this method to explore the parameter
        space around thosee best-fit values.

        Check lmfit doc for more informations.
        """
        try:

            y = self.used_df[y_axis].values[i:j]
            x = self.used_df[x_axis].values[i:j]

            self.mi = self.out
            self.mi.params.add('__lnsigma', value=np.log(
                0.1), min=np.log(0.001), max=np.log(2))
            self.resi = self.mod.fit(
                y, params=self.mi.params, x=x, method="emcee", nan_policy='omit')

            plt.plot(self.resi.acceptance_fraction)
            plt.xlabel('walker')
            plt.ylabel('acceptance fraction')
            plt.title("Rule of thumb, should be between 0.2 and 0.5.")
            plt.show()
            plt.close()

            self.emcee_plot = corner.corner(
                self.resi.flatchain,
                labels=self.resi.var_names,
                truths=list(self.resi.params.valuesdict().values())
            )
            plt.savefig(
                f"{self.folders[3]}{self.used_dataset.filename}_corner_plot.pdf")
            plt.savefig(
                f"{self.folders[3]}{self.used_dataset.filename}_corner_plot.png")

            print('median of posterior probability distribution')
            print('--------------------------------------------')
            lmfit.report_fit(self.resi.params)

            self.p = self.pars.copy()
            used_param = self.resi.var_names
            del used_param[-1]

            highest_prob = np.argmax(self.resi.lnprob)
            hp_loc = np.unravel_index(highest_prob, self.resi.lnprob.shape)
            mle_soln = self.resi.chain[hp_loc]
            for i, par in enumerate(used_param):
                self.p[par].value = mle_soln[i]

            print('\nMaximum Likelihood Estimation from emcee       ')
            print('-------------------------------------------------')
            print('Parameter  MLE Value   Median Value   Uncertainty')
            fmt = '  {:5s}  {:11.5f} {:11.5f}   {:11.5f}'.format
            for name, param in self.p.items():
                if self.resi.params[name].stderr:
                    print(fmt(
                        name,
                        param.value,
                        self.resi.params[name].value,
                        self.resi.params[name].stderr
                    ))

            print('\nError Estimates from emcee    ')
            print('------------------------------------------------------')
            print('Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma ')

            for name, param in self.p.items():
                if self.resi.params[name].stderr:
                    quantiles = np.percentile(self.resi.flatchain[name],
                                              [2.275, 15.865, 50, 84.135, 97.275])
                    median = quantiles[2]
                    err_m2 = quantiles[0] - median
                    err_m1 = quantiles[1] - median
                    err_p1 = quantiles[3] - median
                    err_p2 = quantiles[4] - median
                    fmt = '  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format
                    print(fmt(name, err_m2, err_m1, median, err_p1, err_p2))

        except (AttributeError, KeyError):
            plt.close()
            if y_axis == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")
        except Exception as e:
            raise e

    # Plotting interactive function

    def plot_dataset(
        self,
        spec_number,
        plot_df,
        x,
        y,
        x_axis,
        y_axis,
        title,
        check_plot
    ):
        """
        Allows one to plot one Dataset or all spectra together
        and to then save the figure
        """

        if check_plot == "Plot" and len(spec_number) == 1:
            # Bokeh plot
            try:
                # Create source
                df = getattr(spec_number[0], plot_df)

                source = ColumnDataSource(
                    data=dict(
                        x=df[x],
                        y=df[y]
                    ))

                # Create figure
                TOOLTIPS = [
                    (f"{x_axis} (eV), {y_axis}", "($x, $y)"),
                    ("index", "$index"),
                ]

                p = figure(
                    height=self.figure_height, width=self.figure_width,
                    tools="xpan, pan, wheel_zoom, box_zoom, reset, undo, redo, crosshair, hover, save",
                    tooltips=TOOLTIPS,
                    x_axis_location="above",
                    title=title,
                    x_range=(df[x].values[0], df[x].values[-1]),
                    x_axis_label=x_axis + "(eV)",
                    y_axis_label=y_axis,
                    active_scroll="wheel_zoom",
                )

                p.line("x", "y", source=source,
                       legend_label=spec_number[0].filename.split("/")[-1])

                # Create second figure
                select = figure(
                    height=150, width=self.figure_width,
                    title="Select range here",
                    toolbar_location=None,
                    x_range=(df[x].values[0], df[x].values[-1]),
                )

                # Create range tool
                range_tool = RangeTool(x_range=p.x_range)
                range_tool.overlay.fill_color = "navy"
                range_tool.overlay.fill_alpha = 0.2
                select.add_tools(range_tool)
                select.toolbar.active_multi = range_tool
                select.yaxis.axis_label = y_axis
                select.xaxis.axis_label = x_axis + "(eV)"

                # Add line
                select.line("x", "y", source=source)

                # Show figure
                show(column(p, select))

            except AttributeError:
                print(
                    f"{spec_number[0].filename} does not have the {plot_df} dataframe associated yet.")

        elif check_plot == "Plot" and len(spec_number) > 1:
            # Create figure
            TOOLTIPS = [
                (f"{x_axis} (eV)", "$x"),
                (y_axis, "$y"),
            ]

            p = figure(
                height=self.figure_height, width=self.figure_width,
                tools="xpan, pan, wheel_zoom, box_zoom, reset, undo, redo, crosshair, hover, save",
                tooltips=TOOLTIPS,
                title=title,
                x_axis_label=x_axis,
                y_axis_label=y_axis,
                active_scroll="wheel_zoom"
            )

            p.add_layout(Legend(), 'right')

            # Count number of scans with good df
            nb_color = 0

            # Iterate on class list
            for j, C in enumerate(spec_number):
                try:
                    df = getattr(C, plot_df)

                    # Get attr for legend
                    try:
                        scan = int(C.filename[6:11])
                        row = self.scan_table[self.scan_table.Scan == scan]

                        if self.legend == "condition":
                            legend = f"{scan}, {row.Condition.values[0]}"
                        elif self.legend == "edge":
                            legend = f"{scan}, {row.Edge.values[0]}"
                        elif self.legend == "scan":
                            legend = scan
                        else:
                            legend = f"{scan}, {row.Condition.values[0]}, {row.Edge.values[0]}"
                    except:
                        legend = C.filename

                    # Create source
                    source = ColumnDataSource(
                        data=dict(
                            x=df[x].values,
                            y=df[y].values,
                        ))

                    color = self.matplotlib_colours[nb_color % len(
                        self.matplotlib_colours)]
                    nb_color += 1

                    # Add line
                    p.line(
                        x='x', y='y', source=source, legend_label=legend,
                        line_width=1, line_color=color, line_alpha=0.8,
                        hover_line_color=color, hover_line_alpha=1.0,
                        hover_line_width=2.0, muted_alpha=0.1)

                except AttributeError:
                    pass
                    print(
                        f"The {C.filename} class does not have this DataFrame.")

            if nb_color == 0:
                raise AttributeError(
                    "None of these classes have this DataFrame.")
            else:
                # Show figure
                p.legend.click_policy = "mute"
                show(p)

        elif len(spec_number) == 0:
            print("Select more datasets.")

        else:
            clear_output(True)

    # All handler functions

    def tool_handler(self, change):
        """
        Handles changes on the widget used for the deletion of previous work
        and beginning of data treatment
        """

        if change.new == "default":
            self._list_widgets_init.children[0].disabled = False

        else:
            self._list_widgets_init.children[0].disabled = True

    def reduce_bool_handler(self, change):
        """
        Handles changes on the widget used to decide whether or not we start
        the reduction in the reduction tab.
        """
        if self._list_tab_reduce_method.children[0].value != "Splines":
            if change.new:
                for w in self._list_tab_reduce_method.children[:4]:
                    w.disabled = True
            elif not change.new:
                for w in self._list_tab_reduce_method.children[:4]:
                    w.disabled = False

        elif self._list_tab_reduce_method.children[0].value == "Splines":
            if change.new:
                for w in [
                    self._list_tab_reduce_method.children[0],
                    self._list_tab_reduce_method.children[2],
                    self._list_tab_reduce_method.children[3]
                ]:
                    w.disabled = True
            elif not change.new:
                for w in self._list_tab_reduce_method.children[:4]:
                    w.disabled = False

    def error_extraction_handler(self, change):
        """
        Handles changes on the widget used to extract errors"""

        if change.new:
            for w in self._list_errors_extraction.children[0:7]:
                w.disabled = True
        elif not change.new:
            for w in self._list_errors_extraction.children[0:7]:
                w.disabled = False

    def tools_bool_handler(self, change):
        """
        Handles changes on the widget used to decide whether or not we start
        the data treatment in the tools tab.
        """

        if change.new:
            self.tab_tools.children[0].disabled = True
        elif not change.new:
            self.tab_tools.children[0].disabled = False

    def delimiter_decimal_separator_handler(self, change):
        if change.new != ".npy":
            for w in self._list_import_data.children[2:4]:
                w.disabled = False
        if change.new == ".npy":
            for w in self._list_import_data.children[2:4]:
                w.disabled = True

    def fit_handler(self, change):
        """
        Handles changes on the widget used to pick the dataframe and
        spectra during the fitting routine.
        """

        if change.new:
            self._list_define_fitting_df.children[0].disabled = True
            self._list_define_fitting_df.children[1].disabled = True
        elif not change.new:
            self._list_define_fitting_df.children[0].disabled = False
            self._list_define_fitting_df.children[1].disabled = False

    def model_handler(self, change):
        """
        Handles changes on the widget list after fixing the fitting routine.
        """

        if change.new:
            for w in self._list_define_model.children[:6]:
                w.disabled = True
            for w in self._list_define_model.children[7:10]:
                w.disabled = True
        elif not change.new:
            for w in self._list_define_model.children[:6]:
                w.disabled = False
            for w in self._list_define_model.children[7:10]:
                w.disabled = False

    def model_degree_handler(self, change):
        """
        Handles changes on the widget used to pick the degree of the polynomial
        background in the fitting routine.
        """
        if change.new == PolynomialModel:
            self._list_define_model.children[6].disabled = False
        elif change.new != PolynomialModel:
            self._list_define_model.children[6].disabled = True

    def param_victoreen_handler_single(self, change):
        """
        Handles changes on the widgets used to pick the value of the initial
        parameter of the 1 victoreen function
        """
        if change.new == "victoreen":
            self._list_reduce_single_spline.children[4].disabled = False
            self._list_reduce_single_spline.children[5].disabled = False

        elif change.new != "victoreen":
            self._list_reduce_single_spline.children[4].disabled = True
            self._list_reduce_single_spline.children[5].disabled = True

    def param_victoreen_handler_1(self, change):
        """
        Handles changes on the widgets used to pick the value of the initial
        parameter of the 1 victoreen function
        """
        if change.new == "victoreen":
            self._list_reduce_splines.children[5].disabled = False
            self._list_reduce_splines.children[6].disabled = False

        elif change.new != "victoreen":
            self._list_reduce_splines.children[5].disabled = True
            self._list_reduce_splines.children[6].disabled = True

    def param_victoreen_handler_2(self, change):
        """
        Handles changes on the widgets used to pick the value of the initial
        parameter of the 2 victoreen function
        """
        if change.new == "victoreen":
            self._list_reduce_splines.children[7].disabled = False
            self._list_reduce_splines.children[8].disabled = False

        elif change.new != "victoreen":
            self._list_reduce_splines.children[7].disabled = True
            self._list_reduce_splines.children[8].disabled = True

    """Additional functions"""

    def victoreen(self, x, A, B):
        """victoreen function"""
        return A*x**(-3) + B*x**(-4)

    def save_df_to_csv(self, dataset, df_name):
        "Save dataframe as csv"
        getattr(dataset, df_name).to_csv(
            self.folders[2] +
            dataset.filename.split(".")[0] + f"_{df_name}.csv",
            header=True, index=False
        )

        print("Saved dataframe.")
