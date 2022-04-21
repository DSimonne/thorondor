import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
import errno
import os
import shutil
import math

import lmfit
from lmfit import minimize, Parameters, Parameter
from lmfit.models import *
import corner
from scipy.stats import chisquare

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import display, Markdown, Latex, clear_output

from scipy import interpolate
from scipy import optimize, signal
from scipy import sparse

from datetime import datetime
import pickle
import inspect

import tables as tb

from thorondor.gui_iterable import Dataset


class Interface():
    """
    This  class is a Graphical User Interface (gui) that is meant to be used
    to process important amount of XAS datasets that focus on the same energy
    range and absoption edge.
    There are two ways of initializing the procedure in a jupyter notebook:
        _ gui = thorondor.gui.Interface(); One will have to write the name of
            the data folder in which all his datasets are saved.
        _ gui = thorondor.gui.Interface.get_class_list(data_folder =
            "<yourdata_folder>") if one has already worked on a Dataset and
            wishes to retrieve his work

    This class makes extensive use of the ipywidgets and is thus meant to
    be used with a jupyter notebook.
    Additional informations are provided in the "ReadMe" tab of the gui.
    """

    def __init__(self, class_list=False):
        """
        All the widgets for the GUI are defined here.
        Two different initialization procedures are possible depending on
        whether or not a class_list is given in entry.
        """

        self.work_dir = "./"
        self.path_elements = inspect.getfile(np).split(
            "__")[0].split("numpy")[0] + "thorondor/Elements/"

        if class_list:

            self.class_list = class_list
            self.data_folder = class_list[0].saving_directory.split(
                "/Classes")[0].split("/")[-1]

            path_original_data = self.work_dir + str(self.data_folder)
            path_classes = path_original_data + "/Classes"
            path_data_as_csv = path_original_data + "/ExportData"
            path_figures = path_original_data + "/Figures"
            path_import_data = path_original_data + "/ImportData"

            self.folders = [path_original_data, path_classes,
                            path_data_as_csv, path_figures, path_import_data]

            path_to_classes = [
                p.replace("\\", "/") for p in sorted(
                    glob.glob(path_classes+"/*.pickle")
                )
            ]
            self.names = [
                "Dataset_"+f.split("/")[-1].split(".")[0]
                for f in path_to_classes
            ]

            self.new_energy_column = np.round(
                self.class_list[0].df["Energy"].values, 2)
            self.interpol_step = np.round(
                self.new_energy_column[1] - self.new_energy_column[0], 2)

            # Take shifts into account
            self.new_energy_column = np.linspace(
                self.new_energy_column[0]-20,
                self.new_energy_column[-1]+20,
                int(((self.new_energy_column[-1]+20) -
                    (self.new_energy_column[0]-20))/self.interpol_step + 1))

        elif not class_list:
            self.class_list = []

            self.new_energy_column = np.round(np.linspace(0, 1000, 2001), 2)
            self.interpol_step = 0.05

        # Widgets for the initialization
        self._list_widgets_init = interactive(
            self.class_list_init,
            data_folder=widgets.Text(
                value="data_folder",
                placeholder='<yourdatafolder>',
                description="Data folder:",
                disabled=False,
                style={'description_width': 'initial'}),
            fix_name=widgets.Checkbox(
                value=False,
                description='Fix the name of the folder.',
                disabled=False,
                style={'description_width': 'initial'}),
            create_folders=widgets.Checkbox(
                value=False,
                description='Create/check subdirectories for the program.',
                disabled=True,
                style={'description_width': 'initial'},
                layout=Layout(width="70%")),
            data_type=widgets.Dropdown(
                options=[
                    ".txt", ".dat", ".csv", ".xlsx", ".nxs"],
                value=".txt",
                description='Data type:',
                disabled=True,
                style={'description_width': 'initial'}),
            delimiter_type=widgets.Dropdown(
                options=[
                    ("Comma", ","),
                    ("Tabulation", "\t"),
                    ("Semicolon", ";"),
                    ("Space", " ")
                ],
                value="\t",
                description='Column delimiter type:',
                disabled=True,
                style={'description_width': 'initial'}),
            decimal_separator=widgets.Dropdown(
                options=[
                    ("Dot", "."), ("Comma", ",")],
                value=".",
                description='Decimal delimiter type:',
                disabled=True,
                style={'description_width': 'initial'}),
            marker=widgets.Checkbox(
                value=False,
                description="Initial and final markers",
                disabled=True,
                style={'description_width': 'initial'}),
            initial_marker=widgets.Text(
                value="BEGIN",
                placeholder='<initial_marker>',
                description="Initial marker:",
                disabled=True,
                style={'description_width': 'initial'}),
            final_marker=widgets.Text(
                value="END",
                placeholder='<final_marker>',
                description="Final marker:",
                disabled=True,
                style={'description_width': 'initial'}),
            delete=widgets.Checkbox(
                value=False,
                description='Delete all data and reset work !',
                disabled=True,
                style={'description_width': 'initial'}),
            work=widgets.Checkbox(
                value=False,
                description='Start working !',
                disabled=True,
                style={'description_width': 'initial'}))

        self._list_widgets_init.children[1].observe(
            self.name_handler, names="value")
        self._list_widgets_init.children[2].observe(
            self.create_handler, names="value")
        self._list_widgets_init.children[3].observe(
            self.excel_handler, names="value")
        self._list_widgets_init.children[6].observe(
            self.marker_handler, names="value")
        self._list_widgets_init.children[9].observe(
            self.delete_handler, names="value")
        self._list_widgets_init.children[10].observe(
            self.work_handler, names="value")

        self.tab_init = widgets.VBox([
            widgets.HBox(self._list_widgets_init.children[:2]),
            self._list_widgets_init.children[2],
            widgets.HBox(self._list_widgets_init.children[3:6]),
            widgets.HBox(self._list_widgets_init.children[6:9]),
            widgets.HBox(self._list_widgets_init.children[9:11]),
            self._list_widgets_init.children[-1]
        ])

        # Widgets for the data visualisation
        self._list_data = interactive(
            self.print_data,
            spec=widgets.Dropdown(
                options=self.class_list,
                description='Select the Dataset:',
                disabled=True,
                style={'description_width': 'initial'},
                layout=Layout(width='60%')),
            printed_df=widgets.Dropdown(
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
            show=widgets.Checkbox(
                value=False,
                description='Show dataframe',
                disabled=True,
                style={'description_width': 'initial'}))
        self._list_data.children[2].observe(
            self.show_data_handler, names="value")

        self.tab_data = widgets.VBox([
            self._list_data.children[0],
            widgets.HBox(self._list_data.children[1:3]),
            self._list_data.children[-1]
        ])

        # Widgets for the tools
        self.tab_tools = interactive(
            self.treat_data,
            method=widgets.ToggleButtons(
                options=[
                    ("Flip", "flip"),
                    ("Stable Monitor Norm.", "stable_monitor"),
                    ("Relative shifts correction", "relative_shift"),
                    ("Global shift correction", "global_shift"),
                    ("Gas correction", "gas"),
                    ("Membrane correction", "membrane"),
                    ("Deglitching", "deglitching"),
                    ("Merge energies", "merge"),
                    ("Determine errors", "errors"),
                    ("Import data", "import"),
                    ("Linear Combination Fit", "LCF"),
                    ("Save as .nxs (NeXuS)", "nexus"),
                ],
                value="relative_shift",
                description='Tools:',
                disabled=True,
                button_style="",
                tooltips=[
                    'Correct the possible energy shifts between datasets',
                    "Correct a global energy shift",
                    'Correct for gas absorption',
                    "Correct for membrane absorption",
                    "Deglitch alien points",
                    "Merge datasets together and export as csv",
                    "Determine errors, see ReadMe"
                ],
                style={'description_width': 'initial'}),
            plot_bool=widgets.Checkbox(
                value=False,
                description='Fix tool',
                disabled=True,
                style={'description_width': 'initial'}))
        self.tab_tools.children[1].observe(
            self.tools_bool_handler, names="value")

        self._list_flip = interactive(
            self.flip_axis,
            spec_number=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[0:1],
                rows=5,
                description='Spectra to correct:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column')),
            df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines")
                ],
                value="df",
                description='Use the dataframe:',
                disabled=False,
                style={'description_width': 'initial'}),
            x=widgets.Dropdown(
                options=["Energy"],
                value="Energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("Weights", "weights"),
                    ("RMS", "RMS"),
                    ("User error", "user_error")
                ],
                value="value",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            shift=widgets.FloatText(
                step=0.1,
                value=1,
                description='Shift (a. u.):',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_flip = widgets.VBox([
            self._list_flip.children[0],
            widgets.HBox(self._list_flip.children[1:4]),
            self._list_flip.children[-2],
            self._list_flip.children[-1]
        ])

        self._list_stable_monitor = interactive(
            self.stable_monitor_method,
            spec_number=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[0:1],
                rows=5,
                description='Spectra to correct:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column')),
            df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines")
                ],
                value="df",
                description='Use the dataframe:',
                disabled=False,
                style={'description_width': 'initial'}),
            sample_intensity=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC")
                ],
                value="value",
                description='Select the sample data',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="50%")),
            reference_intensity=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Mesh", "mesh"),
                    ("Reference first normalization", "reference_first_norm"),
                    ("Reference shift", "reference_shift")
                ],
                value="value",
                description='Select the reference.',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="50%")),
            compute=widgets.Checkbox(
                value=False,
                description='Compute ratio sample int./ref. int.',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="80%")),)
        self.widget_list_stable_monitor = widgets.VBox([
            self._list_stable_monitor.children[0],
            self._list_stable_monitor.children[1],
            widgets.HBox(self._list_stable_monitor.children[2:4]),
            self._list_stable_monitor.children[-2],
            self._list_stable_monitor.children[-1]
        ])

        self._list_relative_shift = interactive(
            self.relative_energy_shift,
            spec=widgets.Dropdown(
                options=self.class_list,
                description='reference spectra :',
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
                    ("Energy", "Energy")],
                value="Energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("RMS", "RMS"),
                ],
                value="value",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            fix_ref=widgets.Checkbox(
                value=False,
                description='Fix reference spectra',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_relative_shift = widgets.VBox([
            self._list_relative_shift.children[0],
            widgets.HBox(self._list_relative_shift.children[1:4]),
            self._list_relative_shift.children[4],
            self._list_relative_shift.children[-1]
        ])
        self._list_relative_shift.children[4].observe(
            self.relative_shift_bool_handler, names="value")

        self._list_global_shift = interactive(
            self.global_energy_shift,
            spec_number=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[0:1],
                rows=5,
                description='Spectra to correct:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column')),
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
                options=["Energy"],
                value="Energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("Weights", "weights"),
                    ("RMS", "RMS"),
                    ("User error", "user_error")
                ],
                value="value",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            shift=widgets.FloatText(
                step=self.interpol_step,
                value=0,
                description='Shift (eV):',
                readout=True,
                readout_format='.2f',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_global_shift = widgets.VBox([
            self._list_global_shift.children[0],
            widgets.HBox(self._list_global_shift.children[1:4]),
            self._list_global_shift.children[-2],
            self._list_global_shift.children[-1]
        ])

        self._list_correction_gas = interactive(
            self.correction_gas,
            spec_number=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[0:1],
                rows=5,
                description='Spectra to correct:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column')),
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
                options=["Energy"],
                value="Energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                ],
                value="value",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            gas=widgets.Text(
                value="""{"He":1, "%":100}""",
                description='Gas.es:',
                disabled=False,
                continuous_update=False,
                style={'description_width': 'initial'}),
            d=widgets.FloatText(
                step=0.0001,
                value=0.0005,
                description='membrane thickness:',
                disabled=False,
                style={'description_width': 'initial'}),
            p=widgets.FloatText(
                value=101325,
                description='Pressure:',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_correction_gas = widgets.VBox([
            self._list_correction_gas.children[0],
            widgets.HBox(self._list_correction_gas.children[1:4]),
            widgets.HBox(self._list_correction_gas.children[4:7]),
            self._list_correction_gas.children[-1]
        ])

        self._list_correction_membrane = interactive(
            self.correction_membrane,
            spec_number=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[0:1],
                rows=5,
                description='Spectra to correct:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column')),
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
                options=["Energy"],
                value="Energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                ],
                value="value",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            apply_all=widgets.Checkbox(
                value=False,
                description='Combine gas & Membrane correction.',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_correction_membrane = widgets.VBox([
            self._list_correction_membrane.children[0],
            widgets.HBox(self._list_correction_membrane.children[1:4]),
            self._list_correction_membrane.children[-2],
            self._list_correction_membrane.children[-1]
        ])

        self._list_deglitching = interactive(
            self.correction_deglitching,
            spec=widgets.Dropdown(
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
                options=["Energy"],
                value="Energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("Weights", "weights"),
                    ("RMS", "RMS"),
                    ("User error", "user_error")
                ],
                value="value",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            tipo=widgets.Dropdown(
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

        self._list_merge_energies = interactive(
            self.merge_energies,
            spec_number=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[0:1],
                rows=5,
                description='Spectra to merge:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column')),
            df=widgets.Dropdown(
                options=[
                    ("Renamed data", "df"),
                    ("Shifted data", "shifted_df"),
                    ("Reduced data", "reduced_df"),
                    ("Reduced by Splines", "reduced_df_splines"),
                    ("Fitted data", "fit_df")
                ],
                value="reduced_df",
                description='Use the dataframe:',
                disabled=False,
                style={'description_width': 'initial'}),
            x=widgets.Dropdown(
                options=["Energy"],
                value="Energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("Weights", "weights"),
                    ("RMS", "RMS"),
                    ("User error", "user_error")
                ],
                value="value",
                description='Pick an y-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            title=widgets.Text(
                value="<newcsvfile>",
                placeholder="<newcsvfile>",
                description='Type the name you wish to save:',
                disabled=False,
                continuous_update=False,
                style={'description_width': 'initial'}),
            merge_bool=widgets.Checkbox(
                value=False,
                description='Start merging',
                disabled=False,
                style={'description_width': 'initial'}))
        self._list_merge_energies.children[5].observe(
            self.merge_bool_handler, names="value")
        self.widget_list_merge_energies = widgets.VBox([
            self._list_merge_energies.children[0],
            widgets.HBox(self._list_merge_energies.children[1:4]),
            self._list_merge_energies.children[-3],
            self._list_merge_energies.children[-2],
            self._list_merge_energies.children[-1]
        ])

        self._list_errors_extraction = interactive(
            self.errors_extraction,
            spec=widgets.Dropdown(
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
            xcol=widgets.Dropdown(
                options=["Energy"],
                value="Energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            ycol=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("Weights", "weights"),
                ],
                value="value",
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
            spec=widgets.Dropdown(
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
                options=["Energy"],
                value="Energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
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

        self._list_import_data = interactive(
            self.import_data,
            data_name=widgets.Text(
                placeholder='<name>',
                description='Name:',
                disabled=False,
                continuous_update=False,
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            data_format=widgets.Dropdown(
                options=[(".npy"), (".csv"), (".txt"), (".dat"), (".nxs")],
                value=".npy",
                description='Format:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            delimiter_type=widgets.Dropdown(
                options=[
                    ("Comma", ","), ("Tabulation", "\t"),
                    ("Semicolon", ";"), ("Space", " ")],
                value="\t",
                description='Column delimiter type:',
                disabled=True,
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            decimal_separator=widgets.Dropdown(
                options=[("Dot", "."), ("Comma", ",")],
                value=".",
                description='Decimal delimiter type:',
                disabled=True,
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            energy_shift=widgets.FloatText(
                value=0,
                step=self.interpol_step,
                description='Energy shift (eV):',
                readout=True,
                readout_format='.2f',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')),
            scale_factor=widgets.FloatText(
                step=0.01,
                value=1,
                description='Scale factor:',
                readout=True,
                readout_format='.2f',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(width="50%", height='40px')))
        self.widget_list_import_data = widgets.VBox([
            widgets.HBox(self._list_import_data.children[:2]),
            widgets.HBox(self._list_import_data.children[2:4]),
            widgets.HBox(self._list_import_data.children[4:6]),
            self._list_import_data.children[-1]
        ])
        self._list_import_data.children[1].observe(
            self.delimiter_decimal_separator_handler, names="value")

        self._list_save_as_nexus = interactive(
            self.save_as_nexus,
            spec_number=widgets.SelectMultiple(
                options=self.class_list,
                value=self.class_list[0:1],
                rows=5,
                description='Spectra to save:',
                disabled=False,
                style={'description_width': 'initial'},
                layout=Layout(display="flex", flex_flow='column')),
            apply_all=widgets.Checkbox(
                value=False,
                description='Save',
                disabled=False,
                style={'description_width': 'initial'}))
        self.widget_list_save_as_nexus = widgets.VBox([
            widgets.HBox(self._list_save_as_nexus.children[:2]),
            self._list_save_as_nexus.children[-1]
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
            used_datasets=widgets.Dropdown(
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

        self.tab_reduce_method = widgets.VBox([self._list_tab_reduce_method.children[0], self._list_tab_reduce_method.children[1], self._list_tab_reduce_method.children[2], widgets.HBox(
            self._list_tab_reduce_method.children[3:5]), self._list_tab_reduce_method.children[-1]])

        # Widgets for the LSF background reduction and normalization method
        self._list_reduce_LSF = interactive(
            self.reduce_LSF,
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="value",
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
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="value",
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
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="value",
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
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="value",
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
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="value",
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
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                ],
                value="value",
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
            spec=widgets.Dropdown(
                options=self.class_list,
                description='Select the Dataset:',
                disabled=True,
                style={'description_width': 'initial'},
                layout=Layout(width='60%')),
            printed_df=widgets.Dropdown(
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
            show=widgets.Checkbox(
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
            xcol=widgets.Dropdown(
                options=["Energy"],
                value="Energy",
                description='Pick an x-axis',
                disabled=False,
                style={'description_width': 'initial'}),
            ycol=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                ],

                value="\u03BC",
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
                    ("Energy", "Energy")],
                value="Energy",
                description='Pick an x-axis',
                disabled=True,
                style={'description_width': 'initial'}),
            y=widgets.Dropdown(
                options=[
                    ("Select a value", "value"),
                    ("Sample intensity", "sample_intensity"),
                    ("\u03BC", "\u03BC"),
                    ("Mesh", "mesh"),
                    ("Reference shift", "reference_shift"),
                    ("First normalized \u03BC", "first_normalized_\u03BC"),
                    ("Gas corrected", "gas_corrected"),
                    ("Membrane corrected", "membrane_corrected"),
                    ("Gas & membrane corrected", "gas_membrane_corrected"),
                    ("Background corrected", "background_corrected"),
                    ("Second normalized \u03BC", "second_normalized_\u03BC"),
                    ("Fit", "fit"),
                    ("Weights", "weights"),
                    ("RMS", "RMS"),
                    ("User error", "user_error")
                ],
                value="\u03BC",
                description='Pick an y-axis',
                disabled=True,
                style={'description_width': 'initial'}),
            x_axis=widgets.Text(
                value="Energy",
                placeholder="Energy",
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
                options=[('Clear', "Zero"), ('Plot', "Plot"), ("3D", "3D")],
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

        # Widgets for the logbook
        self._list_print_logbook = interactive(
            self.print_logbook,
            logbook_name=widgets.Text(
                value="logbook.xlsx",
                placeholder='<logbookname>.xlsx',
                description='Type the name of the logbook:',
                disabled=True,
                continuous_update=False,
                style={'description_width': 'initial'}),
            logbook_bool=widgets.Checkbox(
                value=True,
                description='Reset and hide logbook',
                disabled=True,
                style={'description_width': 'initial'}),
            column=widgets.Text(
                value="Quality",
                placeholder='Quality',
                description='Column:',
                continuous_update=False,
                disabled=True,
                style={'description_width': 'initial'}),
            value=widgets.Text(
                value="True",
                placeholder='True',
                description='Value:',
                continuous_update=False,
                disabled=True,
                style={'description_width': 'initial'}))
        self.tab_logbook = widgets.VBox([
            widgets.HBox(self._list_print_logbook.children[:2]),
            widgets.HBox(self._list_print_logbook.children[2:4]),
            self._list_print_logbook.children[-1]
        ])

        # Widgets for the ReadMe
        self.tab_readme = interactive(
            self.display_readme,
            contents=widgets.ToggleButtons(
                options=[
                    'Treatment',
                    'Reduction',
                    'Fit',
                    "Else"
                ],
                value="Treatment",
                description='Show info about:',
                disabled=False,
                tooltips=[
                    'Nothing is shown',
                    'Insight in the functions used for treatment',
                    'Insight in the functions used for Background',
                    'Insight in the functions used for fitting'
                ],
                style={'description_width': 'initial'}))

        # Create the final window
        self.window = widgets.Tab(children=[
            self.tab_init,
            self.tab_data,
            self.tab_tools,
            self.tab_reduce_method,
            self.tab_fit,
            self.tab_plot,
            self.tab_logbook,
            self.tab_readme
        ])
        self.window.set_title(0, 'Initialize')
        self.window.set_title(1, 'View Data')
        self.window.set_title(2, 'Tools')
        self.window.set_title(3, 'Reduce Background')
        self.window.set_title(4, 'Fit')
        self.window.set_title(5, 'Plot')
        self.window.set_title(6, 'Logbook')
        self.window.set_title(7, 'Readme')

        # Display window
        if class_list:
            self._list_widgets_init.children[0].value = self.data_folder
            self._list_widgets_init.children[1].value = True
            self._list_widgets_init.children[2].value = True
            self._list_widgets_init.children[10].value = False

            for w in self._list_widgets_init.children[:-2]:
                w.disabled = True

            for w in self._list_data.children[:-1]:
                w.disabled = False

            for w in self.tab_tools.children[:-1]:
                w.disabled = False

            for w in self._list_tab_reduce_method.children[:-1]:
                w.disabled = False

            for w in self._list_define_fitting_df.children[:-1]:
                w.disabled = False

            for w in self._list_plot_dataset.children[:-1]:
                w.disabled = False

            for w in self._list_print_logbook.children[:-1]:
                w.disabled = False

            # Show the plotting first
            self.window.selected_index = 5

            display(self.window)

        elif not class_list:
            display(self.window)

    # Readme interactive function
    def display_readme(self, contents):
        """
        All the necessary information to be displayed via the ReadMe tab
        are written here in a Markdown format.
        """

        if contents == "Treatment":
            clear_output(True)
            display(Markdown("""## Citation and additional details:

                THORONDOR: software for fast treatment and analysis of low-energy XAS dat. Simonne, D.H., Martini, A.,
                Signorile, M., Piovano, A., Braglia, L., Torelli, P., Borfecchia, E. & Ricchiardi, G. (2020).
                J. Synchrotron Rad. 27, https://doi.org/10.1107/S1600577520011388.
                """))

            display(Markdown("""<strong>IMPORTANT NOTICE</strong>"""))
            display(Markdown("""This program only considers a common energy range to all spectra, if you work on another energy range, please create 
                by hand a new folder, put your data there, and create a new notebook dedicated to this energy range that works on this data_folder.
                Basically, one notebook works on one folder where all the data files have a common energy range (and incrementation). Create different 
                data_folders if you work on different absorption edges !"""))
            display(Markdown("""Throughout your work, the docstring of each function is available for you by creating a new cell with the `plus` button 
                at the top of your screen and by typing: `help(function)`. The detail of the gui can be accessed by `help(thorondor.gui.Interface)`."""))
            display(Markdown("""The classes that are continuously saved and that can be reimported through are instances of the Dataset class.
                For details, please type : `help(thorondor.Dataset)`"""))

            display(Markdown("""## Dataframes"""))
            display(Markdown("""Throughout the program, you will use several dataframes that act as checkpoints in the analysis of your data. They are 
                each created as the output of specific methods.
                The `Renamed data` dataframe is created after you import the data and rename its columns. It is the first checkpoint, you should work
                on this dataframe during the treatment methods such as gas correction, Membrane correction, deglitching or stable monitor normalization.
                The `Shifted data` dataframe is created after you used one of the energy shifting methods, you should perform these two methods on the 
                `Renamed data` dataframe after having treated the data.
                You then possess a dataframe with the data shifted and treated.
                The `Reduced data` dataframe is created after you used one of the background reduction method. You should use these methods on the `Shifted data` dataframe,
                you then possess a dataframe with the treated, energy shifted and background corrected data. The `Reduced by Splines` dataframe is specific to the use of 
                the splines methods since one may want to try and compare several background reduction methods.
                The `Fit data` dataframe is the output of the fitting routine and allows to subsequently plot your fits and compare them.
                Whenever a DataFrame is created or modified, a corresponding .csv file is also created or modified in a subdirectory."""))

            display(Markdown("""## Basic informations"""))
            display(Markdown("""The raw data files from the instrument you must save in the "data_folder" folder that you specify in entry. 
                Please create the folder and move the files inside. If there is a problem, open one file and verify the delimiter and type of the data.
                The notebook needs to be in the same directory as the data_folder containing your datafiles."""))
            display(Markdown("""There is a known issue when importing data files that have a comma instead of a dot as decimal separator, please use a
                dot ! Modify in the initialisation tab if needed."""))
            display(Markdown("""All the datasets that you create are saved in <data_folder>/Data as .csv files.
                The figures in <data_folder>/Figures."""))
            display(Markdown("""Code to automatically expand the jupyter cells :

                `%%javascript
                IPython.OutputArea.prototype._should_scroll = function(lines) {
                    return false;
                }`"""))
            display(Markdown("""Possible problems :
                If you have two different types of files in your folder (e.g. when there are markers in one file and no markers in another, the initialisation will fail."""))

            display(Markdown("""### Interpolation"""))
            display(Markdown("""Interpolating the data is recommended if you have a different amount of points and/or a different energy range between the different datasets.
                We use the scipy technique involving splines, details can be found here : https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
                There is no smoothing of the data points."""))

            display(Markdown("""## 1) Extract Data"""))
            display(Markdown("""$I_s$, $I_m$ and $I_t$ are extracted from the experimental data. These notations follow the work environment of APE-He, the 
                ambient pressure soft x-ray absorption spectroscopy beamline at Elettra, where the NEXAFS spectra are recorded in Transmission Electron Yield (TEY).
                Two electrical contacts allow us to polarize the membrane (positively in order to accelerate the electrons away from the sample), and to 
                measure the drain current from the sample through a Keithley 6514 picoammeter (Rev. Sci. Instrum. 89, 054101 (2018)).
                $I_s$ is the intensity recorded from the sample, $I_m$ is the mesh intensity (used to normalize the intensity collected from the membrane. Indeed, a different photon flux 
                covers the membrane at different energy) and $I_t$ is the quotient of both. In the gui, $I_t$ is renamed \u03BC.
                The absorption spectrum is usually acquired by moving the monochromator with a discrete step and recording the TEY intensity at this energy, 
                repeating this operation for the entire range of interest.
                During the continuous or “fast scan” data acquisition mode, the grating monochromator is scanned continuously through the wanted energy range
                and the picoammeter signal is recorded in the streaming mode.
                If you do not have computed \u03BC prior to the importation of the data, the program will perform the computation automatically by dividing  the column $I_s$ by the 
                column mesh"""))

            display(Markdown("""## 2) Energy shifts"""))
            display(Markdown("""You must find the shift by analysing a reference feature in the reference column of your Dataset. For users of APE-He, the Mesh
             or Keithley both provide static features that allow you to align your datasets. First, choose the reference Dataset to which all the other datasets
             will be shifted, and use the cursor to pick the reference point on this Dataset. 
             The shift is then computed as the difference between the energy that corresponds to the reference cursor and the energy that corresponds to the cursor 
             that you chose on the other datasets. The new data frame is saved as \"shifted_df\" and also as a \"<name>_Shifted.csv\" file.
             it is possible to apply a global shifts to the datasets, e.g. if the position of the peaks are known from literature."""))

            display(Markdown("""## 3) Transmittance Correction"""))
            display(Markdown("""When measuring the drain current in the reactor cell (both from the sample and the membrane), we need to take into account that
             several sources of electrons are present due to the absorption of the primary beam by the reactor cell membrane, the reactant gas, and the sample.
             This effects have been described in Castán-Guerrero, C., Krizmancic, D., Bonanni, V., Edla, R., Deluisa, A., Salvador, F., Rossi, G., Panaccione,
              G., & Torelli, P. (2018). A reaction cell for ambient pressure soft x-ray absorption spectroscopy. Review of Scientific Instruments, 89(5).
              https://doi.org/10.1063/1.5019333 .
              The additional terms to the TEY intensity from the sample can be considered constant as due to constant cross sections, assuming that there are
               no absorption edges of elements present in the membrane or in the gas in the scanned energy range.
              This gui provides corrections for the X-ray transmittance factors linked to both the gas inside the cell and the membrane."""))

            display(Markdown("""### 3.1) Control Parameters:"""))
            display(Markdown("""The parameters $d$ and $p$ represent, respectively, the gas thickness and the total gas pressure in the cell. Each gas that is
                in the cell must be written as a dictionnary, with each element next to its stochiometry and the total percentage of this gas in the cell. The 
                primary interaction of low-energy x rays within matter, viz. photoabsorption and coherent scattering, have been described for photon energies 
                outside the absorption threshold regions by using atomic scattering factors, $f = f_1 + i f_2$. The atomic photoabsorption cross section,
                $µ_a$ may be readily obtained from the values of $f_2$ using the following relation,
                $μ_a = 2r_0 λf_2$
                where $r_0$ is the classical electron radius, and $λ$ is the wavelength. The transmission of x rays through a slab of thickness $d$ is
                then given by,
                $T = \exp (- n \,µ_a d)$
                where n is the number of atoms per unit volume in the slab. In a first approach:
                $n = \\frac{p}{k_b T}$
                ref : http://henke.lbl.gov/optical_constants/intro.html"""))
            display(Markdown(
                """<strong>Please respect the following architecture when using this function:</strong>"""))
            display(Markdown(
                """E.g.: `{"C":1, "O":1, "%":60}, {"He":1, "%":20}, {"O":1, "%":20}`"""))
            display(Markdown("""You may put as many gas as you want, the sum of their percentage must be equal to 1 !The values of $d$, the membrane 
                thickness and $p$, the gas pressure can change but respect the units (meters and Pascals). In the left pannel is reported the imaginary 
                part of the scattering factor $f_{2}$ plotted vs the energy for the different elements involved in the gas feed composition.
                On the right pannel is reported the transmittance trend of the gas feed for each spectra. To change the temperature, you must associate the
                entries in the logbook tab."""))

            display(
                Markdown("""### 3.2)  Transmittance correction for the membrane:"""))
            display(Markdown(
                """The correction due to the membrane is based on the $Si_3 N_4$ membrane used at APE-He in Elettra."""))

            display(Markdown(""" ## 4)  Deglitching"""))
            display(Markdown("""Once that the glitch region has been isolated, three kind of functions $(linear, quadratic, cubic)$ can be used 
                to remove it. By pressing the button "Deglitch" the modication is directely saved. The "deglitching" routine returns the degliched data."""))

        if contents == "Reduction":
            clear_output(True)
            display(Markdown("""### Case n° 1: Least Squares"""))
            display(Markdown("""This kind of data normalization form is based on the the "Asymmetric Least Squares Smooting" technique. 
                More information about it can be found in : "Baseline Correction with Asymmetric Least SquaresSmoothing" 
                of Paul H. C. Eilers and Hans F.M. Boelens. The background removal procedure is regulated by two parameters: 
                $p$ and $\lambda$. The first parameter: p regulates the asymmetry of the background and it can be moved in the 
                following range: $0.001\leq p \leq 0.1$   while $\lambda$ is a smoothing parameter that can vary between $10^{2}$ 
                and $10^{9}$."""))

            display(Markdown(""" ### Case n°2 : Chebyshev polynomials"""))
            display(Markdown("""First kind Chebyshev polynomialsTn, especially for high temperatures.  The weights as well as the degree N of the equation
            were found empirically. The weights taken during the weighed least squares regression are simply taken as the square of the variance of the 
            counting statistics. The background is then given by:
                $f(x, \\vec{a}) = \sum_{n=0}^N a_n T_n(x) \, + \, \epsilon$
                with $a_n$ the $N+ 1$ coefficients determined by the weighed least square regression."""))

            display(Markdown("""### Case n°3: Polynoms"""))
            display(Markdown(""" The "Polynoms" function allows user to perform the data subtraction using the `splrep` method of the SciPy package (Virtanen).
             Once that the user fixed the energy range of interest and the amount of slider points, each point on the related curve can be moved through sliders.
             Two or more point can not take the same value. In this case the background curve is not plotted."""))

            display(
                Markdown("""### Case n°4: Data Reduction and Normalization with Splines"""))

            display(Markdown("""#### 1) E0 selection:"""))
            display(Markdown("""We must first fix the edge jump "E0" for each Dataset ! Attention, if there is a pre-edge, it is possible that the 
                finding routine will fit on it instead than on the edje-jump. Please readjust manually in that case."""))

            display(Markdown("""#### 2) Data Normalization:"""))
            display(Markdown("""The data is normalised by the difference between the value of both polynomials at the point E0. 
                Away from edges, the energy dependence fits a power law: $µ \\sim AE^{-3}+BE^{-4}$ (Victoreen)"""))

        if contents == "Fit":
            clear_output(True)
            display(Markdown("""### Process modelling"""))
            display(Markdown("""Process modelling is defined as the description of a <strong>response variable</strong> $y$ by the summation of a deterministic component 
                given by a <strong>mathematical function</strong> $f(\\vec{x},\\vec{\\beta})$ plus a random error $\\epsilon$ that follows its own probability distribution 
                (see the engineering statistics handbook published by the National Institute of Standards and Technology). """))
            display(
                Markdown("""We have: $\\quad y = f(\\vec{x};\\vec{\\beta}) + \\epsilon$"""))
            display(Markdown("""Since the model cannot be solely equaled to the data by the deterministic mathematical function $f$, we talk of statistical model that are only relevant for the 
                average of a set of points y. Each response variable $\\vec{y_i}$ defined by the model is binned to a predictor variable $\\vec{x_i}$ which are inputs to the 
                mathematical function. $\\vec{\\beta}$ is the set of parameters that will be used and refined during the modelling process."""))
            display(Markdown("""In general we have:"""))
            display(
                Markdown("""$\\quad \\vec{x} \\equiv (x_1, x_2,..., x_N),$"""))
            display(
                Markdown("""$\\quad \\vec{y} \\equiv (y_1, y_2,..., y_N),$"""))
            display(Markdown(
                """$\\quad \\vec{\\beta} \\equiv (\\beta_1,\\beta_2, ..., \\beta_M)$"""))
            display(Markdown("""It is important to differentiate between errors and residuals, if one works with a sample of a population and evaluates the deviation between one element of the 
                sample and the average value in the sample, we talk of residuals. However, the error is the deviation between the value of this element and the the average on the 
                whole population, the true value that is unobservable. For least squares method, the residuals will be evaluated, difference between the observed value and the 
                mathematical function."""))
            display(Markdown("""The value of the parameters is usually unknown before modelling unless for simulation experiments where one uses a model with a predetermined set of 
                parameters to evaluate its outcome. For refinement, the parameters can be first-guessed and approximated from literature (e.g. the edge jump) but it is the 
                purpose of the refinement to lead to new and accurate parameters. The relation between the parameters and the predictor variables depends on the nature of our problem."""))
            display(Markdown("""### Minimization process"""))
            display(Markdown("""The "method of least squares" that is used to obtain parameter estimates was independently developed in the late 1700's and the early 1800's by the 
                mathematicians Karl Friedrich Gauss, Adrien Marie Legendre and (possibly) Robert Adrain [Stigler (1978)] [Harter (1983)] [Stigler (1986)] working in Germany, France 
                and America, respectively."""))
            display(Markdown("""To find the value of the parameters in a linear or nonlinear least squares method, we use the weighed least squares method. 
                The function $f(\\vec{x}, \\hat{\\vec{\\beta}})$ is fitted to the data $y$ by minimizing the following criterion:"""))
            display(Markdown(
                """$\chi^2 = \sum_{i=0}^N W_i \\big( y_i-f(x_i; \\hat{\\vec{\\beta}}) \\big)^2 = \\sum_{i=0}^N W_i \, r_i^2$"""))
            display(Markdown(
                """with N the amount of ($\\theta_i, y_i$) bins in our experiment and $r_i$ the residuals."""))
            display(Markdown("""The sum of the square of the deviations between the data point $y_i$ of the ($\\theta_i, y_i$) bin and the corresponding $f(\\vec{x_i}; \\hat{\\vec{\\beta}})$
            in the model is minimized. For nonlinear models such as ours, the computation must be done via iterative algorithms. The algorithm finds the solution of a system in which each 
            of the partial derivative with respect to each parameter is zero, i.e. when the gradient is zero."""))
            display(Markdown(
                """Documentation for lmfit can be found here : https://lmfit.github.io/lmfit-py/intro.html"""))

            display(Markdown("""### Fitting guidelines"""))
            display(Markdown("""In general, a NEXAFS spectrum is always characterized by resonances corresponding to different transitions from an occupied core
                state to an unfilled final state (Gann et al., 2016). These resonances can be usually modelled as peak shapes, properly reproduced by Lorentzian
                peak-functions (de Groot, 2005; Henderson et al., 2014; Stöhr, 1992; Watts et al., 2006). The procedure of peak decomposition becomes extremely
                important when one wants to decompose a NEXAFS spectrum into a set of peaks where each of them can be assigned to an existing electronic transition.
                Finally, spectral energy shifts for a set of scans can be recovered from the fitting procedure too, they correspond to the inflexion point in the
                ionization potential energy step function (i.e. the maximum of their first derivatives). The evaluation of these quantities is extremely important
                because they properly indicate the presence of reduction or oxidation phenomena involving the system under study, the fitting procedure allows the
                user to extract rigorous mathematical values from the data."""))

            display(Markdown("""thorondor offers a large class of peak functions including Gaussian, Lorentzian, Voight and pseudo-Voigt profiles. The signal ionization potential
                step can be properly modelled using an arc-tangent function (Poe et al., 2004) as well as an error function, also proven suitable for the usage
                (Henderson et al., 2014; Outka & Stohr, 1988). In general, the user should pick a step-function according to his knowledge prior to the fitting,
                since it has been shown that the width of the error function is related to the instrumental resolution (Outka & Stohr, 1988), whereas the width of
                the arc-tangent is related to the life time of the excited state. The step localization depends on the quality of the spectrum, usually several eV
                below the core level ionization energy (Outka & Stohr, 1988). Sometimes, the background in the pre-edge can slightly differ from the step function
                due to features linked to transition to the bound states in the system (de Groot, 2005). In thorondor, if one wishes to focus on that energy range,
                it is possible to use splines of different order to fit the baseline for those energy values and then pass to fit and normalize the pre-edge peaks
                (Wilke et al., 2001)"""))

            display(Markdown(
                """### The basic differences between a Gaussian, a Lorentzian and a Voigt distribution:"""))

            def gaussian(x, amp, cen, wid):
                return amp * np.exp(-(x-cen)**2 / wid)

            x = np.linspace(-10, 10, 101)
            y = gaussian(x, 4.33, 0.21, 1.51) + \
                np.random.normal(0, 0.1, x.size)

            modG = GaussianModel()
            parsG = modG.guess(y, x=x)
            outG = modG.fit(y, parsG, x=x)

            modL = LorentzianModel()
            parsL = modL.guess(y, x=x)
            outL = modL.fit(y, parsL, x=x)

            modV = VoigtModel()
            parsV = modV.guess(y, x=x)
            outV = modV.fit(y, parsV, x=x)

            fig, axs = plt.subplots(figsize=(10, 6))

            # axs.plot(x, y, 'b')
            # axs.plot(x, outG.init_fit, 'k--', label='initial fit')
            axs.plot(x, outG.best_fit, label='Gaussian')

            # axs.plot(x, outL.init_fit, 'k--', label='initial fit')
            axs.plot(x, outL.best_fit, label='Lorentzian')

            # axs.plot(x, outV.init_fit, 'k--', label='initial fit')
            axs.plot(x, outV.best_fit, label='Voigt')
            axs.legend()
            plt.show()

        if contents == "Else":
            clear_output(True)
            display(Markdown("""### Import additional data"""))
            display(Markdown("""To compare your data with simulations from tools like CrisPy (https://www.esrf.eu/computing/scientific/crispy/installation.html), one may
                use the \"import\" tab to import spectra in the gui. It is possible to subsequently launch some basic analysis such as linear combination fits (LCF tab) 
                with the imported data. Note that simulated data usually comes without background, and that you should perform a background reduction routine on your 
                data before comparison."""))

            display(Markdown("""### Logbook"""))
            display(Markdown("""The logbook needs to be in an excel format, please follow : 
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html.
                The logbook also needs to be in the same directory as the data_folder containing your datafiles."""))
            display(Markdown("""The logbook importation routine assumes that the name of each Dataset is stored in a \"Name\" column. The names MUST BE the same names
                as the datasets given in entry to the program. The only other possibility is to have the names preceded of \"scan_\", followed by the Dataset number.
                E.g, for a file \"215215.txt\" given in entry, in the logbook its name is either \"215215\"" or \"scan_215215\"."""))
            display(Markdown("""To allow a better visualization of the data, it is possible to apply a mask or filter to the logbook rows, to reveal the one of interest.
                Supposing that your logbook stores all the Dataset that you have acquired, then you may add a column called \"Quality\" whose value can be True or False, to 
                quickly be able to focus on the Dataset of good quality. The filtering is up to you, and depends on the columns and rows of your logbook."""))
            display(Markdown("""To use the plotting and the gas correction tab to their fullest, one should associate a logbook to the data, with a column named "Temp (K)"
                This will allow the automatic extraction and association of the temperature values to each Dataset."""))

    # Initialization function if previous work had been done
    @staticmethod
    def get_class_list(data_folder):
        """
        Retrieve all the Dataset classes pickled in data_folder

        :param data_folder: folder used to find Dataset
        """
        work_dir = "./"
        path_classes = work_dir + str(data_folder)+"/Classes"
        path_to_classes = [p.replace("\\", "/")
                           for p in sorted(glob.glob(path_classes+"/*.pickle"))]
        names = ["Dataset_" + f.split("/")[-1].split(".")[0]
                 for f in path_to_classes]
        class_list = []

        for n, f in zip(names, path_to_classes):
            try:
                class_list.append(Dataset.unpickle(f))

            except EOFError:
                print(
                    f"{n} is empty, restart the procedure from the beginning, \
                    this may be due to a crash of Jupyter.")
            except FileNotFoundError:
                print(f"The Class does not exist for {n}")

        return Interface(class_list)

    # Initialization interactive function, if no previous work had been done
    def class_list_init(
        self,
        data_folder,
        fix_name,
        create_folders,
        data_type,
        delimiter_type,
        decimal_separator,
        marker,
        initial_marker,
        final_marker,
        delete,
        work
    ):
        """
        Function that generates or updates three subfolders in the "work_dir":
            _ data_folder where you will save your data files.
            _ data_folder/export_data where the raw data files will be saved
              (stripped of all metadata), in .txt format
            _ data_folder/classes where the data will be saved as a Dataset class
              at the end of your work.
            _ data_folder/import_data where you can add extra files used in
              fitting

        :param data_folder: root folder to store data in
        :param fix_name: True to fix data_folder
        :param create_folders: True to create subdirectories
        :param data_type: Data type to be loaded (.csv, .txt, ...)
        :param delimiter_type: Delimiter type in text data
        :param decimal_separator: Decimal separator in text data
        :param marker: True to load the data only between two markers
        :param initial_marker: Initial marker
        :param final_marker: Final marker
        :param delete: Delete all processed data
        :param work: Start working
        """

        if fix_name:
            self.data_folder = data_folder

            path_original_data = self.work_dir + str(self.data_folder)
            path_classes = path_original_data + "/Classes"
            path_data_as_csv = path_original_data + "/ExportData"
            path_figures = path_original_data + "/Figures"
            path_import_data = path_original_data + "/ImportData"

            self.folders = [path_original_data, path_classes,
                            path_data_as_csv, path_figures, path_import_data]

        if fix_name and create_folders:
            clear_output = (True)

            for folder in self.folders:
                if not os.path.exists(folder):
                    try:
                        os.makedirs(folder)
                        print(f"{folder} well created.\n")
                    except FileExistsError:
                        print(f"{folder} already exists.\n")
                    except Exception as e:
                        raise e

        if fix_name and delete:
            self._list_data.children[0].options = []

            self._list_relative_shift.children[0].options = []
            self._list_correction_gas.children[0].options = []
            self._list_correction_membrane.children[0].options = []
            self._list_deglitching.children[0].options = []
            self._list_merge_energies.children[0].options = []
            self._list_errors_extraction.children[0].options = []
            self._list_LCF.children[0].options = []
            self._list_LCF.children[1].options = []
            self._list_LCF.children[2].options = []
            self._list_save_as_nexus.children[0].options = []

            self._list_tab_reduce_method.children[1].options = []
            self._list_tab_reduce_method.children[2].options = []

            self._list_define_fitting_df.children[0].options = []

            self._list_plot_dataset.children[0].options = []

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
            clear_output = (True)

        if not work:
            for w in self._list_data.children[:-1] + \
                    self.tab_tools.children[:-1] + \
                    self._list_tab_reduce_method.children[:-1] + \
                    self._list_define_fitting_df.children[:-1] + \
                    self._list_plot_dataset.children[:-1] + \
                    self._list_print_logbook.children[:-1]:
                if not w.disabled:
                    w.disabled = True

        if work:
            print("""
                We now start to manipulate the data.
                \nFirst, rename each column and select the one we want to use.
                """)

            self.data_type = data_type
            self.file_locations = [p.replace("\\", "/")
                                   for p in sorted(
                glob.glob(
                    f"{self.folders[0]}/*{self.data_type}")
            )]

            # Possible issues here
            self.names = [
                "Dataset_"+f.split("/")[-1].split(".")[0]
                for f in self.file_locations
            ]

            clear_output(True)

            def renaming(nb_columns, new_name):
                ButtonSaveName = Button(
                    description="Save name",
                    layout=Layout(width='15%', height='35px'))
                ButtonSaveList = Button(
                    description="Show new df",
                    layout=Layout(width='25%', height='35px'))
                display(widgets.HBox((ButtonSaveName, ButtonSaveList)))

                @ButtonSaveName.on_click
                def ActionSaveName(selfbutton):
                    if new_name == "select":
                        print("Please select a value")
                    else:
                        self.newnames[nb_columns] = new_name
                        print(f"Column renamed")
                        print(
                            f"The list of new names is currently {self.newnames}.")

                @ButtonSaveList.on_click
                def ActionSaveList(selfbutton):
                    """
                    Apply the renaming to all the files given in entry only if
                    name given for each column and at least one column names
                    energy
                    """
                    if all([i is not None for i in self.newnames]):
                        usecol = [i for i, j in enumerate(
                            self.newnames) if j != "notused"]
                        namae = [j for i, j in enumerate(
                            self.newnames) if j != "notused"]

                        if "\u03BC" in self.newnames \
                                or "\u03BC" not in self.newnames and "sample_intensity" in self.newnames and "mesh" in self.newnames:

                            try:
                                if self.data_type == ".xlsx":
                                    dataset_renamed = pd.read_excel(
                                        self.file_locations[0], header=0, names=namae,  usecols=usecol).abs()
                                elif self.data_type == ".nxs":
                                    with tb.open_file(self.file_locations[0], "r") as f_nxs:
                                        dataset_renamed = pd.DataFrame(
                                            f_nxs.root.NXentry.NXdata.data[:]).abs()
                                else:
                                    dataset_renamed = pd.read_csv(
                                        self.file_locations[0], sep=delimiter_type, header=0, names=namae,  usecols=usecol, decimal=decimal_separator).abs()

                                dataset_renamed = dataset_renamed.sort_values(
                                    "Energy").reset_index(drop=True)

                                if "\u03BC" not in self.newnames:
                                    dataset_renamed["\u03BC"] = dataset_renamed["sample_intensity"] / \
                                        dataset_renamed["mesh"]

                                    print(
                                        "The \u03BC column was computed as the ratio of the intensity of the signal coming from the sample (Is) over the incident flux beam intensity (mesh)")

                                print("This is the renamed data.")
                                display(dataset_renamed.head())

                                ButtonSaveNameAllNoInterpol = Button(
                                    description="Keep and continue without interpolation.",
                                    layout=Layout(width='35%', height='35px'))
                                ButtonSaveNameAllInterpol = Button(
                                    description="Keep and continue with interpolation.",
                                    layout=Layout(width='35%', height='35px'))

                                # Always with interpolation
                                display(ButtonSaveNameAllInterpol)

                                @ButtonSaveNameAllNoInterpol.on_click
                                def ActionSaveNameAllNoInterpol(selfbutton):
                                    for n, f in zip(self.names, self.file_locations):
                                        try:
                                            if self.data_type == ".xlsx":
                                                dataset_renamed = pd.read_excel(
                                                    f, header=0, names=namae,  usecols=usecol).abs()
                                            elif self.data_type == ".nxs":
                                                with tb.open_file(f, "r") as f_nxs:
                                                    dataset_renamed = pd.DataFrame(
                                                        f_nxs.root.NXentry.NXdata.data[:]).abs()
                                            else:
                                                dataset_renamed = pd.read_csv(
                                                    f, sep=delimiter_type, header=0, names=namae,  usecols=usecol, decimal=decimal_separator).abs()

                                            if "\u03BC" not in self.newnames:
                                                dataset_renamed["\u03BC"] = dataset_renamed["sample_intensity"] / \
                                                    dataset_renamed["mesh"]

                                            dataset_renamed = dataset_renamed.sort_values(
                                                "Energy").reset_index(drop=True)
                                            dataset_renamed.to_csv(
                                                f"{self.folders[2]}/{n}_renamed.csv", index=False)
                                            dataset_renamed["Energy"] = np.round(
                                                dataset_renamed["Energy"], 2)

                                        except TypeError:
                                            # In case one Dataset has a different decimal separator
                                            if decimal_separator == ".":
                                                other_decimal_separator = ","
                                            elif decimal_separator == ",":
                                                other_decimal_separator = "."

                                            dataset_renamed = pd.read_csv(
                                                f, sep=delimiter_type, header=0, names=namae,  usecols=usecol, decimal=other_decimal_separator).abs()

                                            if "\u03BC" not in self.newnames:
                                                dataset_renamed["\u03BC"] = dataset_renamed["sample_intensity"] / \
                                                    dataset_renamed["mesh"]

                                            dataset_renamed = dataset_renamed.sort_values(
                                                "Energy").reset_index(drop=True)
                                            dataset_renamed.to_csv(
                                                f"{self.folders[2]}/{n}_renamed.csv", index=False)
                                            dataset_renamed["Energy"] = np.round(
                                                dataset_renamed["Energy"], 2)

                                        except pd.errors.ParserError:
                                            print(
                                                "Are the names of your files not consistent ?")
                                        except Exception as e:
                                            raise e

                                        finally:
                                            try:
                                                C = Dataset(
                                                    dataset_renamed, f, n, self.folders[1])
                                                self.class_list.append(C)
                                                C.pickle()

                                            except Exception as e:
                                                print(
                                                    f"The class could not been instanced for {n}\n")
                                                raise e

                                    print(
                                        "All data files have also been \
                                        corrected for the negative values of \
                                        the energy and for the possible \
                                        flipping of values.\n")
                                    ButtonSaveNameAllNoInterpol.disabled = True
                                    ButtonSaveNameAllInterpol.disabled = True

                                    for w in self._list_data.children[:-1] + self.tab_tools.children[:-1] + self._list_tab_reduce_method.children[:-1] + self._list_define_fitting_df.children[:-1] + self._list_plot_dataset.children[:-1] + self._list_print_logbook.children[:-1]:
                                        if w.disabled:
                                            w.disabled = False

                                    # Does not update automatically sadly
                                    self._list_data.children[0].options = self.class_list
                                    self._list_flip.children[0].options = self.class_list
                                    self._list_stable_monitor.children[0].options = self.class_list
                                    self._list_relative_shift.children[0].options = self.class_list
                                    self._list_global_shift.children[0].options = self.class_list
                                    self._list_correction_gas.children[0].options = self.class_list
                                    self._list_correction_membrane.children[0].options = self.class_list
                                    self._list_deglitching.children[0].options = self.class_list
                                    self._list_merge_energies.children[0].options = self.class_list
                                    self._list_errors_extraction.children[0].options = self.class_list
                                    self._list_LCF.children[0].options = self.class_list
                                    self._list_LCF.children[1].options = self.class_list
                                    self._list_LCF.children[2].options = self.class_list
                                    self._list_save_as_nexus.children[0].options = self.class_list

                                    self._list_tab_reduce_method.children[1].options = self.class_list
                                    self._list_tab_reduce_method.children[2].options = self.class_list

                                    self._list_define_fitting_df.children[0].options = self.class_list

                                    self._list_plot_dataset.children[0].options = self.class_list

                                def interpolate_data(step_value, interpool_bool):
                                    """
                                    We interpolate the data between the minimum
                                    and maximum energy point common to all the
                                    datasets with the given step
                                    """

                                    if interpool_bool:
                                        Emin, Emax = [], []

                                        # Initialize the class Dataset for each file given in entry
                                        for n, f in zip(self.names, self.file_locations):
                                            try:

                                                if self.data_type == ".xlsx":
                                                    dataset_renamed = pd.read_excel(
                                                        f, header=0, names=namae,  usecols=usecol).abs()
                                                elif self.data_type == ".nxs":
                                                    with tb.open_file(f, "r") as f_nxs:
                                                        dataset_renamed = pd.DataFrame(
                                                            f_nxs.root.NXentry.NXdata.data[:]).abs()
                                                else:
                                                    dataset_renamed = pd.read_csv(
                                                        f, sep=delimiter_type, header=0, names=namae,  usecols=usecol, decimal=decimal_separator).abs()

                                                if "\u03BC" not in self.newnames:
                                                    dataset_renamed["\u03BC"] = dataset_renamed["sample_intensity"] / \
                                                        dataset_renamed["mesh"]

                                                dataset_renamed = dataset_renamed.sort_values(
                                                    "Energy").reset_index(drop=True)
                                                dataset_renamed.to_csv(
                                                    f"{self.folders[2]}/{n}_renamed.csv", index=False)
                                                dataset_renamed["Energy"] = np.round(
                                                    dataset_renamed["Energy"], 2)

                                            except TypeError:
                                                # In case one Dataset has a different decimal separator
                                                if decimal_separator == ".":
                                                    other_decimal_separator = ","
                                                elif decimal_separator == ",":
                                                    other_decimal_separator = "."

                                                dataset_renamed = pd.read_csv(
                                                    f, sep=delimiter_type, header=0, names=namae,  usecols=usecol, decimal=other_decimal_separator).abs()

                                                if "\u03BC" not in self.newnames:
                                                    dataset_renamed["\u03BC"] = dataset_renamed["sample_intensity"] / \
                                                        dataset_renamed["mesh"]

                                                dataset_renamed = dataset_renamed.sort_values(
                                                    "Energy").reset_index(drop=True)
                                                dataset_renamed.to_csv(
                                                    f"{self.folders[2]}/{n}_renamed.csv", index=False)
                                                dataset_renamed["Energy"] = np.round(
                                                    dataset_renamed["Energy"], 2)

                                            except pd.errors.ParserError:
                                                print(
                                                    "Are the names of your files not consistent ? You may have more columns in some files as well.")
                                            except Exception as e:
                                                raise e

                                            finally:
                                                # Store all min and max energy values
                                                Emin.append(
                                                    min(dataset_renamed["Energy"].values))
                                                Emax.append(
                                                    max(dataset_renamed["Energy"].values))

                                                # Append the datasets in class_list
                                                try:

                                                    C = Dataset(
                                                        dataset_renamed, f, n, self.folders[1])
                                                    self.class_list.append(C)
                                                    C.pickle()

                                                except Exception as e:
                                                    print(
                                                        f"The class could not been instanced for {n}\n")
                                                    raise e

                                        # Create a new, common, energy column
                                        self.interpol_step = step_value
                                        self.new_energy_column = np.round(
                                            np.arange(np.max(Emin), np.min(Emax), step_value), 2)

                                        # Interpolation happens here
                                        try:
                                            # Iterate over all the datasets, we drop the duplicate value that could mess up the splines computation
                                            for C in self.class_list:
                                                self.used_df_init = getattr(
                                                    C, "df").drop_duplicates("Energy")

                                                x = self.used_df_init["Energy"]

                                                interpolated_df = pd.DataFrame({
                                                    "Energy": self.new_energy_column
                                                })

                                                # Iterate over all the columns
                                                for col in self.used_df_init.columns[self.used_df_init.columns != "Energy"]:

                                                    y = self.used_df_init[col].values

                                                    tck = interpolate.splrep(
                                                        x, y, s=0)
                                                    y_new = interpolate.splev(
                                                        self.new_energy_column, tck)
                                                    interpolated_df[col] = y_new

                                                setattr(
                                                    C, "df", interpolated_df)
                                                C.pickle()

                                        except Exception as e:
                                            raise e

                                        print(
                                            "All data files have also been corrected for the negative values of the energy and for the flipping of values.\n")
                                        ButtonSaveNameAllNoInterpol.disabled = True
                                        ButtonSaveNameAllInterpol.disabled = True

                                        self._list_interpol.children[0].disabled = True
                                        self._list_interpol.children[1].disabled = True

                                        for w in self._list_data.children[:-1] + self.tab_tools.children[:-1] + self._list_tab_reduce_method.children[:-1] + self._list_define_fitting_df.children[:-1] + self._list_plot_dataset.children[:-1] + self._list_print_logbook.children[:-1]:
                                            if w.disabled:
                                                w.disabled = False

                                        # Does not update automatically sadly
                                        self._list_data.children[0].options = self.class_list

                                        self._list_flip.children[0].options = self.class_list
                                        self._list_stable_monitor.children[0].options = self.class_list
                                        self._list_relative_shift.children[0].options = self.class_list
                                        self._list_global_shift.children[0].options = self.class_list
                                        self._list_correction_gas.children[0].options = self.class_list
                                        self._list_correction_membrane.children[0].options = self.class_list
                                        self._list_deglitching.children[0].options = self.class_list
                                        self._list_merge_energies.children[0].options = self.class_list
                                        self._list_errors_extraction.children[0].options = self.class_list
                                        self._list_LCF.children[0].options = self.class_list
                                        self._list_LCF.children[1].options = self.class_list
                                        self._list_LCF.children[2].options = self.class_list
                                        self._list_save_as_nexus.children[0].options = self.class_list

                                        self._list_tab_reduce_method.children[1].options = self.class_list
                                        self._list_tab_reduce_method.children[2].options = self.class_list

                                        self._list_define_fitting_df.children[0].options = self.class_list

                                        self._list_plot_dataset.children[0].options = self.class_list

                                        # Change displayed energy range selection in case the shifts are important
                                        self.new_energy_column = np.round(np.linspace(self.new_energy_column[0]-20, self.new_energy_column[-1]+20, int(
                                            ((self.new_energy_column[-1]+20) - (self.new_energy_column[0]-20))/self.interpol_step + 1)), 2)

                                        for w in [self._list_reduce_LSF.children[1], self._list_reduce_chebyshev.children[1], self._list_reduce_polynoms.children[1], self._list_reduce_splines_derivative.children[1], self._list_define_model.children[2]]:
                                            w.min = self.new_energy_column[0]
                                            w.value = [
                                                self.new_energy_column[0], self.new_energy_column[-1]]
                                            w.max = self.new_energy_column[-1]
                                            w.step = self.interpol_step

                                    else:
                                        print("Window cleared")

                                @ButtonSaveNameAllInterpol.on_click
                                def ActionSaveNameAllInterpol(selfbutton):
                                    self._list_interpol = interactive(
                                        interpolate_data,
                                        step_value=widgets.FloatSlider(
                                            value=0.05,
                                            min=0.01,
                                            max=0.5,
                                            step=0.01,
                                            description='Step:',
                                            disabled=False,
                                            continuous_update=False,
                                            orientation="horizontal",
                                            readout=True,
                                            readout_format='.2f',
                                            style={'description_width': 'initial'}),
                                        interpool_bool=widgets.Checkbox(
                                            value=False,
                                            description='Fix step',
                                            disabled=False,
                                            style={'description_width': 'initial'}))
                                    self.little_tab_interpol = widgets.VBox([widgets.HBox(
                                        self._list_interpol.children[:2]), self._list_interpol.children[-1]])
                                    display(self.little_tab_interpol)

                            except KeyError:
                                print(
                                    "At least one column must be named \"energy\"")

                            except ValueError:
                                print("Duplicate names are not allowed.")

                        else:
                            print(
                                "You must have either a column named \u03BC or two columns named Is and Mesh to be able to continue.")

                    else:
                        print("Rename all columns before.")

            try:
                if not marker:
                    if self.data_type == ".xlsx":
                        df = pd.read_excel(self.file_locations[0]).abs()
                    elif self.data_type == ".nxs":
                        with tb.open_file(self.file_locations[0], "r") as f_nxs:
                            df = pd.DataFrame(
                                f_nxs.root.NXentry.NXdata.data[:]).abs()
                    else:
                        df = pd.read_csv(
                            self.file_locations[0], sep=delimiter_type, decimal=decimal_separator).abs()

                if marker:
                    # The file needs to be rewriten
                    # Open all the files individually and saves its content in a variable
                    # Only on txt, csv or dat files
                    for file in self.file_locations:
                        with open(file, "r") as f:
                            lines = f.readlines()

                        # Assign new name to future file
                        complete_name = file.split(".txt")[0] + f"~.dat"
                        with open(complete_name, "w") as g:
                            # Save the content between the markers
                            for row in lines:
                                if str(initial_marker)+"\n" in row:
                                    inizio = lines.index(row)

                                if str(final_marker)+"\n" in row:
                                    fine = lines.index(row)

                            for j in np.arange(inizio+1, fine):
                                g.write(lines[j])
                    self.file_locations = [
                        p.replace("\\", "/") for p in sorted(glob.glob(self.folders[0]+"/*~.dat"))]

                    df = pd.read_csv(
                        self.file_locations[0], sep=delimiter_type, header=None, decimal=decimal_separator).abs()

                display(df.head())

                nb_columns_in_data = len(df.iloc[0, :])
                if nb_columns_in_data == 4:
                    self.newnames = [
                        "Energy", "sample_intensity", "mesh", "\u03BC"]
                elif nb_columns_in_data == 2:
                    self.newnames = ["Energy", "\u03BC"]
                elif nb_columns_in_data == 3:
                    self.newnames = ["Energy", "mesh", "\u03BC"]
                elif nb_columns_in_data == 5:
                    self.newnames = ["Energy", "sample_intensity",
                                     "mesh", "\u03BC", "reference_shift"]
                elif nb_columns_in_data == 8:
                    self.newnames = ["Energy", "sample_intensity", "mesh",
                                     "\u03BC", "notused", "notused", "notused", "notused"]
                else:
                    self.newnames = [None] * nb_columns_in_data

                _list_work = interactive(
                    renaming,
                    nb_columns=widgets.Dropdown(
                        options=list(
                            range(nb_columns_in_data)),
                        value=0,
                        description='Column:',
                        disabled=False,
                        style={'description_width': 'initial'}),
                    new_name=widgets.Dropdown(
                        options=[
                            ("Select a value", "value"),
                            ("Energy", "Energy"),
                            ("Sample intensity", "sample_intensity"),
                            ("\u03BC", "\u03BC"),
                            ("Mesh", "mesh"),
                            ("Reference shift", "reference_shift"),
                            ("Reference first normalization",
                             "reference_first_norm"),
                            ("Not used", "notused"),
                            ("User error", "user_error")
                        ],
                        value="value",
                        description="New name for this column:",
                        disabled=False,
                        style={'description_width': 'initial'}))
                little_tab_init = widgets.VBox(
                    [widgets.HBox(_list_work.children[:2]), _list_work.children[-1]])
                display(little_tab_init)

            # except IndexError:
            #     print("Empty folder")

            # except (TypeError, pd.errors.EmptyDataError, pd.errors.ParserError, UnboundLocalError):
            #     print("Wrong data_type/delimiter/marker ! This may also be due to the use of colon as the decimal separator in your data, refer to ReadMe.")

            except Exception as e:
                raise e

    # Visualization interactive function

    def print_data(self, spec, printed_df, show):
        """
        Displays the pandas.DataFrame associated to each Dataset,
        there are currently 4 different possibilities:
            _ df : Original data
            _ shifted_df : Is one shifts the energy 
            _ reduced_df : If one applies some background reduction or
              normalization method
            _ reduced_df_splines : If one applied the specific Splines
              background reduction and normalization method.
        Each data frame is automatically saved as a .csv file after creation.
        """

        if not show:
            print("Window cleared")
            clear_output(True)

        elif show:
            used_df = getattr(spec, printed_df)
            if len(used_df.columns) == 0:
                print(
                    f"This class does not have the {printed_df} dataframe associated yet.")
            else:
                try:
                    display(used_df)
                except AttributeError:
                    print(f"Wrong Dataset and column combination !")

    # Tools global interactive function

    def treat_data(self, method, plot_bool):
        if method == "flip" and plot_bool:
            display(self.widget_list_flip)
        if method == "stable_monitor" and plot_bool:
            display(self.widget_list_stable_monitor)
        if method == "relative_shift" and plot_bool:
            display(self.widget_list_relative_shift)
        if method == "global_shift" and plot_bool:
            display(self.widget_list_global_shift)
        if method == "gas" and plot_bool:
            display(self.widget_list_correction_gas)
        if method == "membrane" and plot_bool:
            display(self.widget_list_correction_membrane)
        if method == "deglitching" and plot_bool:
            display(self.widget_list_deglitching)
        if method == "merge" and plot_bool:
            display(self.widget_list_merge_energies)
        if method == "errors" and plot_bool:
            display(self.widget_list_errors_extraction)
        if method == "LCF" and plot_bool:
            display(self.widget_list_LCF)
        if method == "import" and plot_bool:
            display(self.widget_list_import_data)
        if method == "nexus" and plot_bool:
            display(self.widget_list_save_as_nexus)
        if not plot_bool:
            print("Window cleared")
            clear_output(True)
            plt.close()

    # Tools interactive sub-functions
    def flip_axis(self, spec_number, df, x, y, shift):
        """
        Allows one to crrect a possible flip of the value around the x axis
        """

        if spec_number:
            try:
                fig, axs = plt.subplots(3, figsize=(16, 16))

                axs[0].set_title('Selected datasets before correction')
                axs[0].set_xlabel("Energy")
                axs[0].set_ylabel('NEXAFS intensity')

                axs[1].set_title('Selected datasets after shift')
                axs[1].set_xlabel("Energy")
                axs[1].set_ylabel('NEXAFS intensity')

                axs[2].set_title(
                    'Selected datasets after shift, absolute values')
                axs[2].set_xlabel("Energy")
                axs[2].set_ylabel('NEXAFS intensity')

                for j, C in enumerate(spec_number):
                    used_df = getattr(C, df).copy()

                    # Plot before correction
                    axs[0].plot(used_df[x], used_df[y])

                    # Plots after correction
                    new_y = used_df[y] - shift

                    axs[1].plot(used_df[x], new_y)
                    axs[2].plot(used_df[x], abs(new_y), label=C.name)

                lines, labels = [], []
                for ax in axs:
                    axLine, axLabel = ax.get_legend_handles_labels()
                    lines.extend(axLine)
                    labels.extend(axLabel)

                fig.tight_layout()

                fig.legend(lines,
                           labels,
                           loc='lower center',
                           borderaxespad=0.1,
                           fancybox=True,
                           shadow=True,
                           ncol=5)
                plt.subplots_adjust(bottom=0.15)
                plt.show()

                ButtonFlip = Button(
                    description="Apply flip",
                    layout=Layout(width='15%', height='35px'))
                display(ButtonFlip)

                @ButtonFlip.on_click
                def ActionButtonFlip(selfbutton):
                    "Apply shifts correction"
                    for j, C in enumerate(spec_number):
                        used_df = getattr(C, df)
                        flip_df = used_df.copy()
                        flip_df[y] = abs(used_df[y]-shift)
                        setattr(C, df, flip_df)

                        # Save work
                        flip_df.to_csv(
                            f"{self.folders[2]}/{C.name}_flipped.csv", index=False)
                        C.pickle()
                    print("Flip well applied and saved in same dataframe.")

            except (AttributeError, KeyError):
                clear_output(True)
                plt.close()
                if y == "value":
                    print("Please select a column.")
                else:
                    print(f"Wrong Dataset and column combination !")

        else:
            clear_output(True)
            plt.close()
            print("You need to select at least one Dataset !")

    def stable_monitor_method(
        self,
        spec_number,
        df,
        sample_intensity,
        reference_intensity,
        compute
    ):
        """
        """

        if spec_number and compute:

            try:
                for j, C in enumerate(spec_number):
                    used_df = getattr(C, df)
                    used_df["First normalized \u03BC"] = used_df[sample_intensity] / \
                        used_df[reference_intensity]

                    # Save work
                    used_df.to_csv(
                        f"{self.folders[2]}/{C.name}_StableMonitorNorm.csv", index=False)
                    C.pickle()

                print(
                    "Data well normalised and saved as First normalized \u03BC, in the same dataframe.")

            except (AttributeError, KeyError, NameError):
                plt.close()
                if sample_intensity == "value":
                    print(
                        "Please select a column for the values of the reference sample.")
                else:
                    print(f"Wrong Dataset and column combination !")

                if reference_intensity == "value":
                    print("Please select a column for the values of the mesh.")
                else:
                    print(f"Wrong Dataset and column combination !")

    def relative_energy_shift(self, spec, df, x, y, fix_ref):
        """Allows one to shift each Dataset by a certain amount k"""

        try:
            ref_df = getattr(spec, df)
            ref_df[x]
            ref_df[y]

            # Initialize list
            self.shifts = [0 for i in range(len(self.class_list))]

            _list_minus_ref_dataset = self.class_list.copy()
            _list_minus_ref_dataset.remove(spec)

            if fix_ref:

                @interact(
                    current_dataset=widgets.Dropdown(
                        options=_list_minus_ref_dataset,
                        description='Current spectra:',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(width="60%")),
                    shift=widgets.FloatText(
                        step=self.interpol_step,
                        value=0,
                        description='Shift (eV):',
                        disabled=False,
                        readout=True,
                        readout_format='.2f',
                        style={'description_width': 'initial'}),
                    interval=widgets.FloatRangeSlider(
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
                        layout=Layout(width="50%", height='40px')),
                    cursor=widgets.FloatSlider(
                        min=self.new_energy_column[0],
                        value=self.new_energy_column[0] + (
                            self.new_energy_column[-1] - self.new_energy_column[0])/2,
                        max=self.new_energy_column[-1],
                        step=self.interpol_step,
                        description='cursor position (eV):',
                        disabled=False,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format='.2f',
                        style={'description_width': 'initial'},
                        layout=Layout(width="50%", height='40px')),)
                def show_area(current_dataset, shift, interval, cursor):
                    """Compute area and compare by means of LSM"""
                    try:
                        dataset_index = self.class_list.index(current_dataset)
                        used_df = getattr(self.class_list[dataset_index], df)

                        # Take new area
                        try:
                            i1 = int(np.where(ref_df[x] == interval[0])[0])
                        except TypeError:
                            i1 = 0

                        try:
                            j1 = int(np.where(ref_df[x] == interval[1])[0])
                        except TypeError:
                            j1 = len(ref_df[x]) - 1

                        try:
                            i2 = int(
                                np.where(used_df[x] == interval[0] - shift)[0])
                        except TypeError:
                            i2 = 0

                        try:
                            j2 = int(
                                np.where(used_df[x] == interval[1] - shift)[0])
                        except TypeError:
                            j2 = len(used_df[x]) - 1

                        plt.close()

                        fig, ax = plt.subplots(1, figsize=(16, 8))
                        ax.set_xlabel("Energy")
                        ax.set_ylabel('NEXAFS')
                        ax.set_title(f'{self.class_list[dataset_index].name}')
                        ax.tick_params(direction='in', labelsize=15, width=2)

                        # reference
                        ax.plot(ref_df[x][i1:j1+1], ref_df[y]
                                [i1:j1+1], label='reference Dataset')

                        # Cursor
                        ax.axvline(x=cursor, color='orange', linestyle='--')

                        # Before correction
                        # ax.plot(used_df[x][i2:j2+1], used_df[y][i2:j2+1], label='Before shift', linestyle='--')

                        # After correction
                        ax.plot(used_df[x][i2:j2+1] + shift, used_df[y][i2:j2+1],
                                label=f"Current spectra shifted by {shift} eV.", linestyle='--')
                        ax.legend()

                        ButtonGuessShiftWithFit = widgets.Button(
                            description="Guess shift with LSM.",
                            layout=Layout(width="50%", height='35px'))
                        ButtonGuessShiftWithDerivativeFit = widgets.Button(
                            description="Guess shift with LSM and first order derivative.",
                            layout=Layout(width="50%", height='35px'))
                        ButtonFixShift = widgets.Button(
                            description="Fix shift.",
                            layout=Layout(width="50%", height='35px'))
                        ButtonApplyAllShifts = widgets.Button(
                            description="Apply all the shifts (final step)",
                            layout=Layout(width="50%", height='35px'))
                        display(widgets.VBox((widgets.HBox((ButtonGuessShiftWithFit, ButtonGuessShiftWithDerivativeFit)), widgets.HBox(
                            (ButtonFixShift, ButtonApplyAllShifts)))))

                        @ButtonGuessShiftWithFit.on_click
                        def GuessShiftWithFit(selfbutton):
                            clear_output(True)
                            plt.close()
                            display(widgets.VBox((widgets.HBox((ButtonGuessShiftWithFit, ButtonGuessShiftWithDerivativeFit)), widgets.HBox(
                                (ButtonFixShift, ButtonApplyAllShifts)))))

                            initial_guess = [0]

                            def find_shift(par):
                                tck = interpolate.splrep(
                                    used_df[x][i2:j2 + 1].values + par, used_df[y][i2:j2 + 1].values, s=0)
                                y_new = interpolate.splev(
                                    ref_df[x][i1:j1 + 1].values, tck)

                                return np.sum((ref_df[y][i1:j1 + 1] - y_new) ** 2)

                            LCF_result = optimize.minimize(
                                find_shift, initial_guess)
                            print(
                                "This least-squares routine minimizes the difference between both spectra by working on the shift. A large background difference may impact the final result.")
                            print(LCF_result.message)
                            print(
                                f"Shift value : {(LCF_result.x[0]//self.interpol_step) * self.interpol_step} eV.")
                            print(
                                f"You may adapt the value of the shift according to the previous result if it seems correct. Please use a multiple of {self.interpol_step} eV.")

                        @ButtonGuessShiftWithDerivativeFit.on_click
                        def GuessShiftWithDerivativeFit(selfbutton):
                            clear_output(True)
                            plt.close()
                            display(widgets.VBox((widgets.HBox((ButtonGuessShiftWithFit, ButtonGuessShiftWithDerivativeFit)), widgets.HBox(
                                (ButtonFixShift, ButtonApplyAllShifts)))))

                            initial_guess = [0]

                            def derivative_list(energy, mu):
                                """Return the center point derivative for each point x_i as np.gradient(y) / np.gradient(x)"""
                                dEnergy, dIT = [], []

                                for i in range(len(mu)):
                                    x = energy[i].values
                                    y = mu[i].values

                                    dEnergy.append(x)
                                    dIT.append(np.gradient(y) / np.gradient(x))

                                return dEnergy, dIT

                            dEnergy, dIT = derivative_list(
                                [ref_df[x][i1:j1 + 1], used_df[x][i2:j2 + 1]], [ref_df[y][i1:j1 + 1], used_df[y][i2:j2 + 1]])

                            M = (dEnergy[0][np.where(dIT[0] == min(dIT[0]))[
                                 0][0]]) - (dEnergy[1][np.where(dIT[1] == min(dIT[1]))[0][0]])

                            print(
                                f"The difference between both minimum in the first order derivative is of {M} eV (see figure underneath).")

                            plt.plot(dEnergy[0], dIT[0], label="reference")
                            plt.plot(dEnergy[1], dIT[1],
                                     label="Current spectra")

                            def find_shift_derivative(par):
                                tck = interpolate.splrep(
                                    dEnergy[1] + par, dIT[1], s=0)
                                y_new = interpolate.splev(dEnergy[0], tck)

                                return np.sum((dIT[0] - y_new) ** 2)

                            LCF_result = optimize.minimize(
                                find_shift_derivative, initial_guess)

                            print("This least-squares routine minimizes the difference between both first order spectra derivative by working on the shift. Large variations in the background difference may impact the final result.")
                            print(LCF_result.message)
                            print(f"Shift value : {LCF_result.x[0]} eV.")
                            print(
                                f"You may adapt the value of the shift according to the previous result if it seems correct, or by taking the amount of eV between both minima. Please use a multiple of {self.interpol_step} eV.")

                        @ButtonFixShift.on_click
                        def FixShift(selfbutton):
                            "Fixes the shift"
                            if np.round(shift % self.interpol_step, 2) in [0, self.interpol_step]:
                                self.shifts[dataset_index] = shift
                                print(
                                    f"Shift fixed for Dataset number {current_dataset.name}.\n")
                                print(
                                    f"This list contains the currently stored values for the shifts.\n")
                                for x in self.shifts:
                                    print(x, end=', ')
                                print("\n")
                            else:
                                print(
                                    f"Please use a multiple of {self.interpol_step} eV.")

                        @ButtonApplyAllShifts.on_click
                        def ApplyCorrection(selfbutton):
                            "Apply shifts correction"

                            for C, s in zip(self.class_list, self.shifts):
                                used_df = getattr(C, df)
                                shift_df = used_df.copy()
                                shift_df[x] = np.round(used_df[x]+s, 2)
                                setattr(C, "shifted_df", shift_df)

                                # Save work
                                shift_df.to_csv(
                                    f"{self.folders[2]}/{C.name}_shifted.csv", index=False)
                                C.pickle()
                            print(
                                "Shifts well applied and saved in a new df shifted_df")

                    except (AttributeError, KeyError):
                        plt.close()
                        print(f"Wrong Dataset and column combination !")

            if not fix_ref:
                clear_output(True)
                plt.close()

        except (AttributeError, KeyError):
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")

    def global_energy_shift(self, spec_number, df, x, y, shift):

        if spec_number:
            try:
                fig, axs = plt.subplots(2, figsize=(16, 11))

                axs[0].set_title('Selected datasets before the shift')
                axs[0].set_xlabel("Energy")
                axs[0].set_ylabel('NEXAFS intensity')

                axs[1].set_title('Selected datasets after the shift')
                axs[1].set_xlabel("Energy")
                axs[1].set_ylabel('NEXAFS intensity')

                for j, C in enumerate(spec_number):
                    used_df = getattr(C, df).copy()

                    # Plot before correction
                    axs[0].plot(used_df[x], used_df[y])

                    # Plot after correction
                    axs[1].plot(used_df[x] + shift, used_df[y], label=C.name)

                lines, labels = [], []
                for ax in axs:
                    axLine, axLabel = ax.get_legend_handles_labels()
                    lines.extend(axLine)
                    labels.extend(axLabel)

                fig.tight_layout()

                fig.legend(lines,
                           labels,
                           loc='lower center',
                           borderaxespad=0.1,
                           fancybox=True,
                           shadow=True,
                           ncol=5)
                plt.subplots_adjust(bottom=0.15)
                plt.show()

                ButtonGlobalShift = Button(
                    description="Apply global shift",
                    layout=Layout(width='25%', height='35px'))
                display(ButtonGlobalShift)

                @ButtonGlobalShift.on_click
                def ActionButtonGlobalShift(selfbutton):
                    if np.round(shift % self.interpol_step, 2) in [0, self.interpol_step]:
                        "Apply shifts correction"
                        for j, C in enumerate(spec_number):
                            used_df = getattr(C, df)
                            shift_df = used_df.copy()
                            shift_df[x] += shift
                            setattr(C, "shifted_df", shift_df)
                            # Save work
                            shift_df.to_csv(
                                f"{self.folders[2]}/{C.name}_shifted.csv", index=False)
                            C.pickle()
                        print("Shifts well applied and saved in a new df shifted_df")
                    else:
                        print(
                            f"Please use a multiple of {self.interpol_step} eV.")

            except (AttributeError, KeyError):
                clear_output(True)
                plt.close()
                if y == "value":
                    print("Please select a column.")
                else:
                    print(f"Wrong Dataset and column combination !")

        else:
            clear_output(True)
            plt.close()
            print("You need to select at least one Dataset !")

    def correction_gas(self, spec_number, df, x, y, gas, d, p):
        """
        This function computes the absorbance correction that need to be
        applied due to the presence of gas in the cell.

        Each gas may be imput as a dictionnary and the sum of the percentage of
        each gas must be equal to 100.

        :param d: width of the membrane (unit)
        :param p: gas pressure (unit)
        """
        elements, per = [], []

        try:
            T = [int(C.logbook_entry["Temp (K)"]) for C in spec_number]
            print("The color is function of the temperature for each Dataset.")
        except:
            print(
                "No valid logbook entry for the temperature found as [Temp (K)],\
                 the T for each Dataset will be set to RT.\
                 \nPlease refer to ReadMe.\n")
            T = 273.15 * np.ones(len(spec_number))

        try:
            for j, C in enumerate(spec_number):
                used_df = getattr(C, df)
                used_df[x]
                used_df[y]
            # Make tuple of dict
            gas_list = [i.replace("{", "").replace("}", "")
                        for i in gas.split("},")]
            gas_dict = [dict((k.split(":")[0].strip(" ").strip("\""), int(
                k.split(":")[1])) for k in e.split(",")) for e in gas_list]

            # Make sure that the gazes are well defined
            for g in gas_dict:
                print(g)
                try:
                    if g["%"] < 100 and g["%"] > 0:
                        "Good percentage"
                except (KeyError, TypeError):
                    return "You need to include a valid percentage for"+str(g)+". Please refer to ReadMe."

                if len(g) > 1:
                    per.append(g["%"]/100)
                    del g["%"]
                    if bool(g):
                        elements.append(g)
                else:
                    return "You need to include at least one gas and its stoiechiometry. Please refer to ReadMe."

            if np.sum(per) != 1.0:
                raise ValueError(
                    "The sum of the percentage of the different gazes is not equal to 1! Please refer to ReadMe.")

            # Retrieve the stochiometric number for each atom per gas
            atom = [v for e in elements for k, v in e.items()]
            nb_atoms = [len(e) for e in elements]

            # Variables used
            kb = 1.38064852 * 10 ** (-23)
            # Radius of atom classical approach
            r0 = 2.8179 * 10 ** (-15)
            T = np.array(T)
            # Number of atoms per unit volume in the slab
            n = np.array(p / (kb * T))

            # Name of the files needed in order of the gazes
            nomi = [str(k)+".nff" for e in elements for k, v in e.items()]
            labels = [str(k) for e in elements for k, v in e.items()]

            # Store lambdas, energies and f2, for all the gazes, does not depend on their stochio or %
            energies, lambdas, f2_original = [], [], []
            for file in nomi:
                reference = np.loadtxt(
                    str(self.path_elements) + file, skiprows=1)
                energies.append(reference[:, 0])
                lambdas.append((10 ** (-9)) * (1236) / reference[:, 0])
                f2_original.append(np.transpose(reference[:, 2]))

            # Compute interpolated values for each spectra
            for j, C in enumerate(spec_number):
                used_df = getattr(C, df)
                energy = used_df[x]

                # Real lambda values
                RealL = ((10 ** (-9)) * 1236 / energy)

                f2 = []
                # Interpolate for new f2 values, for all the gases, does not depend on their stochio or %
                for f, e in zip(f2_original, energies):
                    tck = interpolate.splrep(e, f, s=0)
                    f2int = interpolate.splev(energy, tck, der=0)
                    f2.append(np.transpose(f2int))

            # Plotting
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

            # Plot each scattering factor on the instrument's energy range
            axs[0].set_title('Gas scattering Factor')
            for f in f2:
                axs[0].plot(energy, f)
            axs[0].set_xlabel("Energy")
            axs[0].set_ylabel('f2')
            axs[0].legend(labels)
            cmap = matplotlib.cm.get_cmap('Spectral')

            total_transmittance = []
            for unitvolume, t in zip(n, T):
                Tr = []
                count = 0
                for pour, nb in zip(per, nb_atoms):
                    tg = np.ones(len(RealL))
                    for i in range(nb):
                        sto = atom[count + i]
                        tg = tg * np.array([np.exp(-2 * sto * unitvolume * r0 * RealL[j]
                                           * f2[count + i][j] * d) for j in range(len(RealL))])
                    Tr.append(pour * tg)
                    count += nb
                total_transmittance.append(sum(Tr))
                axs[1].plot(energy, sum(Tr), label=f"Total transmittance at T =  {t} K".format(
                    t), color=(t / max(T), 0, (max(T) - t)/max(T)))

            # Put a legend below current axis
            axs[1].legend(loc='upper center', bbox_to_anchor=(
                0, -0.2), fancybox=True, shadow=True, ncol=4)

            axs[1].set_title('Gas Transmittance')
            axs[1].set_xlabel("Energy")
            axs[1].set_ylabel('Transmittance')
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right()
            axs[1].axvline(x=np.min(energy), color='black', linestyle='--')
            axs[1].axvline(x=np.max(energy), color='black', linestyle='--')

            plt.show()

            ButtonApplyGasCorrection = Button(
                description="Apply gas corrections.",
                layout=Layout(width="50%", height='35px'))
            display(ButtonApplyGasCorrection)

            @ButtonApplyGasCorrection.on_click
            def ActionButtonGlobalShift(selfbutton):
                "Apply shifts correction"
                for j, C in enumerate(spec_number):
                    used_df = getattr(C, df)
                    used_df["gas_transmittance"] = total_transmittance[j]
                    used_df["gas_corrected"] = used_df[y] / \
                        used_df["gas_transmittance"]
                    used_df.to_csv(
                        f"{self.folders[2]}/{C.name}_{df}_gas_corrected.csv", index=False)
                    C.pickle()
                    print(f"Correction applied for {C.name}.")

        except (AttributeError, KeyError):
            clear_output(True)
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")
        except ValueError:
            clear_output(True)
            plt.close()
            print("Gases not well defined. Please refer to ReadMe. Please mind as well that all the datasets must be of same length !")
        except UnboundLocalError:
            clear_output(True)
            plt.close()
            print("You need to select at least one class !")

    def correction_membrane(self, spec_number, df, x, y, apply_all):
        """
        Apply the method of correction to the membrane, does not depend on the
        temperature, membrane composition fixed.
        """

        try:
            plt.close()
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.tick_params(direction='in', labelsize=15, width=2)
            ax.set_xlabel("Energy")
            ax.set_ylabel('Transmittance')
            ax.set_xlim(0, 1000)
            ax.set_title('membrane Scattering Factor')

            TM = np.loadtxt(self.path_elements + "membrane.txt")

            # New interpolation
            energyMem = TM[:, 0]
            mem = TM[:, 1]
            tck = interpolate.splrep(energyMem, mem, s=0)

            if spec_number:
                ax.plot(energyMem, mem)

                for j, C in enumerate(spec_number):
                    used_df = getattr(C, df)
                    energy = used_df[x]

                    f2int = interpolate.splev(energy, tck, der=0)

                    used_df["membrane_transmittance"] = f2int
                    used_df["membrane_corrected"] = used_df[y] / \
                        used_df["membrane_transmittance"]
                    print(f"Membrane correction applied to {C.name}")

                # All same energy range so only show the last one
                ax.axvline(x=np.min(energy), color='black', linestyle='--')
                ax.axvline(x=np.max(energy), color='black', linestyle='--')
                ax.plot(energy, f2int)
                plt.show()

            else:
                plt.close()
                print("Please select at least one Dataset.")

            if apply_all:
                for j, C in enumerate(spec_number):
                    try:
                        used_df = getattr(C, df)
                        used_df["membrane_corrected"] = used_df[y] / \
                            used_df["membrane_transmittance"]
                        used_df["total_transmittance"] = used_df["membrane_transmittance"] * \
                            used_df["gas_transmittance"]
                        used_df["gas_membrane_corrected"] = used_df[y] / \
                            used_df["total_transmittance"]
                        used_df.to_csv(
                            f"{self.folders[2]}/{C.name}_{df}_membrane_gas_corrected.csv", index=False)
                        C.pickle()
                        print(
                            f"Gas & membrane corrections combined for {C.name}!")

                    except:
                        print(
                            f"You did not define the gas correction yet for {C.name}!")

        except (AttributeError, KeyError):
            plt.close()
            if y == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")
        except Exception as e:
            raise e

    def correction_deglitching(self, spec, df, pts, x, y, tipo):
        """
        Allows one to delete some to replace glitches in the data by using
        linear, square or cubic interpolation.
        """
        try:
            self.used_datasets = spec
            self.used_df_type = df
            used_df = getattr(self.used_datasets, self.used_df_type)
            used_df[y]

            @interact(
                interval=widgets.IntRangeSlider(
                    value=[len(used_df[x]) // 4, len(used_df[x]) // 2],
                    min=pts,
                    max=len(used_df[x]) - 1 - pts,
                    step=1,
                    description='Range (indices):',
                    disabled=False,
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
                    energy = used_df[x]
                    mu = used_df[y]
                    v1, v2 = interval

                    # Plot
                    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                    axs[0].set_xlabel("Energy")
                    axs[0].set_ylabel('NEXAFS')
                    axs[0].set_title('Raw Data')
                    axs[0].tick_params(direction='in', labelsize=15, width=2)

                    axs[0].plot(energy, mu, label='Data')
                    axs[0].plot(energy[v1:v2], mu[v1:v2], '-o',
                                linewidth=0.2, label='Selected region')

                    axs[0].axvline(x=energy[v1], color='black', linestyle='--')
                    axs[0].axvline(x=energy[v2], color='black', linestyle='--')
                    axs[0].legend()

                    axs[1].set_title('Region Zoom')
                    axs[1].set_xlabel("Energy")
                    axs[1].set_ylabel('NEXAFS')
                    axs[1].tick_params(direction='in', labelsize=15, width=2)

                    axs[1].plot(energy[v1:v2], mu[v1:v2], 'o', color='orange')
                    axs[1].plot(energy[v1 - pts:v1],
                                mu[v1 - pts:v1], '-o', color='C0')
                    axs[1].plot(energy[v2:v2 + pts],
                                mu[v2:v2 + pts], '-o', color='C0')

                    axs[1].yaxis.set_label_position("right")
                    axs[1].yaxis.tick_right()

                    # Interpolate
                    energy_range_1 = energy[v1 - pts:v1]
                    energy_range_2 = energy[v2:v2 + pts]
                    intensity_range_1 = mu[v1 - pts:v1]
                    intensity_range_2 = mu[v2:v2 + pts]
                    energy_range = np.concatenate(
                        (energy_range_1, energy_range_2), axis=0)
                    intensity_range = np.concatenate(
                        (intensity_range_1, intensity_range_2), axis=0)

                    Enew = energy[v1:v2]
                    f1 = interpolate.interp1d(
                        energy_range, intensity_range, kind=tipo)
                    ITN = f1(Enew)

                    axs[1].plot(Enew, ITN, '--', color='green',
                                label='New line')
                    axs[1].legend()

                    ButtonDeglitch = widgets.Button(
                        description="Deglich",
                        layout=Layout(width='25%', height='35px'))
                    display(ButtonDeglitch)

                    @ButtonDeglitch.on_click
                    def ActionButtonDeglitch(selfbutton):
                        C = self.used_datasets
                        used_df = getattr(C, df)
                        used_df[y][v1:v2] = ITN
                        clear_output(True)
                        C.pickle()
                        print(
                            f"Degliched {C.name}, energy Range: [{energy[v1]}, {energy[v2]}] (eV)")
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

    def merge_energies(self, spec_number, df, x, y, title, merge_bool):
        """
        The output of this function is an excel like file (.csv) that is saved
        in the subdirectory data.
        In this single file, the same column in saved for each Dataset on the
        same energy range to ease the plotting outside this gui.
        """
        if merge_bool and len(spec_number) > 1:
            try:
                # Create a df that spans the entire energy range and check if filename if valid
                self.merged_values = pd.DataFrame({
                    x: self.new_energy_column
                })

                self.merged_values.to_csv(
                    f"{self.folders[2]}/{title}.csv", sep=";", index=False, header=True)

            except OSError:
                print("Please specify a new name.")

            try:
                # Merge
                for j, C in enumerate(spec_number):
                    used_df = getattr(C, df).copy()
                    yvalues = pd.DataFrame(
                        {x: used_df[x].values, y: used_df[y].values})

                    for v in self.merged_values[x].values:
                        if v not in yvalues[x].values:
                            yvalues = yvalues.append({x: v}, ignore_index=True).sort_values(
                                by=[x]).reset_index(drop=True)

                    self.merged_values[str(C.name) + "_" + str(y)] = yvalues[y]

                    print(f"{j+1} out of {len(spec_number)} datasets processed.\n")

                self.merged_values.to_csv(
                    f"{self.folders[2]}/{title}.csv", sep=";", index=False, header=True)

                print(
                    f"Datasets merged for {df} and {y}. Available as {title}.csv in the subfolders.")

            except (AttributeError, KeyError):
                plt.close()
                if y == "value":
                    print("Please select a column.")
                else:
                    print(f"Wrong Dataset and column combination !")

            except Exception as e:
                raise e

        elif merge_bool and len(spec_number) < 2:
            plt.close()
            print("Please select at least two datasets.")

        else:
            plt.close()
            print("Window cleared.")
            clear_output(True)

    def errors_extraction(self, spec, df, xcol, ycol, nbpts, deg, direction, compute):

        def poly(x, y, deg):
            coef = np.polyfit(x, y, deg)
            # Create the polynomial function from the coefficients
            return np.poly1d(coef)(x)

        if compute:
            try:
                clear_output(True)
                self.used_datasets, self.used_df_type = spec, df
                used_df = getattr(self.used_datasets, self.used_df_type)
                x = used_df[xcol]
                y = used_df[ycol]

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
                axs[0].set_ylabel('NEXAFS')
                axs[0].set_title('Data')
                axs[0].plot(x, y, label="Data")
                axs[0].legend()

                axs[1].set_xlabel("Energy")
                axs[1].set_ylabel('NEXAFS')
                axs[1].set_title('Root Mean Square')
                axs[1].plot(x, self.errors["RMS"], label="RMS")
                axs[1].legend()

                try:
                    used_df["Deviations"] = self.errors["Deviations"]
                    used_df["RMS"] = self.errors["RMS"]
                except:
                    setattr(self.used_datasets, self.used_df_type, pd.concat(
                        [used_df, self.errors], axis=1, sort=False))

                self.used_datasets.pickle()
                display(getattr(self.used_datasets, self.used_df_type))

            except (AttributeError, KeyError):
                plt.close()
                if ycol == "value":
                    print("Please select a column.")
                else:
                    print(f"Wrong Dataset and column combination !")

            except Exception as e:
                raise e

        else:
            plt.close()
            print("Window cleared.")
            clear_output(True)

    def LCF(self, ref_spectra, spec_number, spec, df_type, x, y, LCF_bool):
        if LCF_bool and len(ref_spectra) > 1:
            self.ref_names = [f.name for f in ref_spectra]

            try:
                def align_ref_and_spec(**kwargs):
                    try:
                        # interval for data
                        v1Data, v2Data = [], []
                        for j, C in enumerate(spec_number):
                            used_df = getattr(C, df_type)
                            try:
                                v1Data.append(
                                    int(np.where(used_df["Energy"].values == self.energy_widgets[0].value[0])[0]))
                            except TypeError:
                                v1Data.append(0)

                            try:
                                v2Data.append(
                                    int(np.where(used_df["Energy"].values == self.energy_widgets[0].value[1])[0]))
                            except TypeError:
                                v2Data.append(len(used_df["Energy"].values)-1)

                        # Take data spectrum on interval
                        self.spec_df = [getattr(D, df_type).copy(
                        )[v1:v2] for D, v1, v2 in zip(spec_number, v1Data, v2Data)]

                        self.used_df_LCF = self.spec_df[spec_number.index(
                            spec)]

                        # Import the references
                        self.ref_df = [getattr(f, df_type).copy()
                                       for f in ref_spectra]

                        # Add shifts and scale factors to references
                        shifts = [c.value for c in self.shift_widgets]
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
                                    int(np.where(used_df["Energy"].values == self.energy_widgets[0].value[0])[0]))
                            except TypeError:
                                v1Ref.append(0)

                            try:
                                v2Ref.append(
                                    int(np.where(used_df["Energy"].values == self.energy_widgets[0].value[1])[0]))
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
                            self.used_df_LCF[x], self.used_df_LCF[y], label=spec.name)
                        ax.legend()
                        plt.title("First visualization of the data")
                        plt.show()

                        # Check if all the references and data spectra have the same interval and nb of points
                        good_range_ref = [np.array_equal(
                            df[x].values, self.used_df_LCF[x].values) for df in self.ref_df]
                        good_range_spec = [np.array_equal(
                            df[x].values, self.used_df_LCF[x].values) for df in self.spec_df]

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
                                        used_df[x], used_df[y], label=C.name)
                                    ax.legend()
                                    plt.title("LCF Result")
                                    plt.show()

                                    # Print detailed result
                                    print(
                                        f"The detail of the fitting for {C.name} is the following:")
                                    print(LCF_result)
                                    print(
                                        f"The weights for the references are {ref_weights}")

                                    r_factor = np.sum(
                                        (self.used_df_LCF[y]-sum_ref_weights)**2) / np.sum((self.used_df_LCF[y])**2)
                                    print(f"R factor :{r_factor}")
                                    setattr(C, "ref_R_factor", r_factor)

                                    C.pickle()

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
                                    [C.name for C in spec_number], rotation=90, fontsize=14)

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
                self.shift_widgets = [widgets.FloatText(
                    value=0,
                    step=self.interpol_step,
                    continuous_update=False,
                    readout=True,
                    readout_format='.2f',
                    description=f"Shift for {n}",
                    style={'description_width': 'initial'}) for n in self.ref_names]
                _list_shift_widgets = widgets.HBox(tuple(self.shift_widgets))

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
                    c.description: c for c in self.energy_widgets + self.shift_widgets + self.intensity_factor_widgets}

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

    def import_data(self, data_name, data_format, delimiter_type, decimal_separator, energy_shift, scale_factor):
        "This function is meant to import a simulated spectrum to be used for comparison."

        ButtonShowData = Button(
            description="Show data",
            layout=Layout(width='30%', height='35px'))
        display(ButtonShowData)

        clear_output(True)

        @ButtonShowData.on_click
        def ActionShowData(selfbutton):
            try:
                if data_format == ".npy":
                    simulated_data_frame = pd.DataFrame(
                        np.load(f"{self.folders[4]}/{data_name}.npy"))
                    simulated_data_frame.columns = ["Energy", "\u03BC"]

                if data_format != ".npy":
                    simulated_data_frame = pd.read_csv(
                        self.folders[4] + "/" + data_name + data_format, header=0, sep=delimiter_type, decimal=decimal_separator)

                # Need to rename as well

                # Adjust if needed
                display(simulated_data_frame)
                simulated_data_frame["Energy"] += energy_shift
                simulated_data_frame["\u03BC"] = simulated_data_frame["\u03BC"] * scale_factor

                self.temp_df = simulated_data_frame

                display(self.temp_df)

                fig, ax = plt.subplots(figsize=(16, 6))
                ax.set_xlabel("Energy")
                ax.set_ylabel('NEXAFS')
                ax.plot(self.temp_df["Energy"], self.temp_df["\u03BC"])

                Buttonimport_data = Button(
                    description="Import data",
                    layout=Layout(width='30%', height='35px'))
                display(Buttonimport_data)

                @Buttonimport_data.on_click
                def Actionimport_data(selfbutton):
                    try:
                        # Interpolation
                        self.temp_df = self.temp_df.drop_duplicates("Energy")
                        old_x = self.temp_df["Energy"]
                        old_y = self.temp_df["\u03BC"]
                        tck = interpolate.splrep(old_x, old_y, s=0)

                        new_energy_column_sim = np.round(
                            np.arange(np.min(old_x), np.max(old_x), self.interpol_step), 2)

                        y_new = interpolate.splev(new_energy_column_sim, tck)

                        interpolated_sim_df = pd.DataFrame({
                            "Energy": new_energy_column_sim,
                            "\u03BC": y_new})

                        # Include in gui
                        C = Dataset(interpolated_sim_df,
                                    self.folders[4], data_name, self.folders[1])
                        C.df["First normalized \u03BC"] = C.df["\u03BC"]
                        C.df["background_corrected"] = C.df["\u03BC"]
                        C.df["second_normalized_\u03BC"] = C.df["\u03BC"]
                        C.df["Fit"] = C.df["\u03BC"]

                        C.shifted_df = C.df.copy()

                        C.reduced_df = C.df.copy()

                        C.reduced_df_splines = C.df.copy()

                        C.fit_df = C.df.copy()

                        class_list_names = [D.name for D in self.class_list]
                        if data_name in class_list_names:
                            del self.class_list[class_list_names.index(
                                data_name)]
                            self.class_list.append(C)

                        else:
                            self.class_list.append(C)

                        C.pickle()
                        print("Successfully added the data to the gui.")

                    except Exception as e:
                        print(f"The class could not been instanced \n")
                        raise e

                    # Does not update automatically sadly
                    self._list_data.children[0].options = self.class_list

                    self._list_flip.children[0].options = self.class_list
                    self._list_stable_monitor.children[0].options = self.class_list
                    self._list_relative_shift.children[0].options = self.class_list
                    self._list_global_shift.children[0].options = self.class_list
                    self._list_correction_gas.children[0].options = self.class_list
                    self._list_correction_membrane.children[0].options = self.class_list
                    self._list_deglitching.children[0].options = self.class_list
                    self._list_merge_energies.children[0].options = self.class_list
                    self._list_errors_extraction.children[0].options = self.class_list
                    self._list_LCF.children[0].options = self.class_list
                    self._list_LCF.children[1].options = self.class_list
                    self._list_LCF.children[2].options = self.class_list
                    self._list_save_as_nexus.children[0].options = self.class_list

                    self._list_tab_reduce_method.children[1].options = self.class_list
                    self._list_tab_reduce_method.children[2].options = self.class_list

                    self._list_define_fitting_df.children[0].options = self.class_list

                    self._list_plot_dataset.children[0].options = self.class_list

            except Exception as E:
                raise E
                print("Could not import the data.")

    def save_as_nexus(self, spec_number, apply_all):
        if apply_all:
            for C in spec_number:
                print(f"Saving as {C.name} ... ")
                C.to_nxs()
                print(f"Saved as {C.name}!", end="\n\n")

    # Reduction interactive function
    def reduce_data(self, method, used_class_list, used_datasets, df, plot_bool):
        """Define the reduction routine to follow depending on the Reduction widget state."""
        try:
            self.used_class_list = used_class_list
            self.used_datasets = used_datasets
            self.used_dataset_position = used_class_list.index(used_datasets)
            clear_output(True)

            # Update
            self._list_reduce_LSF.children[0].value = "value"
            self._list_reduce_chebyshev.children[0].value = "value"
            self._list_reduce_polynoms.children[0].value = "value"
            self._list_reduce_splines_derivative.children[0].value = "value"

            try:
                self.used_df_type = df
                used_df = getattr(self.used_datasets, self.used_df_type)

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
            print(f"{used_datasets.name} is not in the list of datasets to reduce.")

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
            df = self.used_df_type

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
            axs[0].set_ylabel('NEXAFS')
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
            axs[1].set_ylabel('NEXAFS')
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
                C = self.used_datasets
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
                print(f"Saved Dataset {C.name}")
                temp_df.to_csv(
                    f"{self.folders[2]}/{C.name}_reduced.csv", index=False)
                C.pickle()

            @ButtonRemoveBackground.on_click
            def ActionRemoveBackground(selfbutton):
                # Substract background to the intensity
                clear_output(True)

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                ax[0].set_title('Background subtracted')
                ax[0].set_xlabel("Energy")
                ax[0].set_ylabel('NEXAFS')
                ax[0].tick_params(direction='in', labelsize=15, width=2)

                ax[1].set_title('Background subtracted shifted')
                ax[1].set_xlabel("Energy")
                ax[1].set_ylabel('NEXAFS')
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
                                   0.1*(i), label=self.used_class_list[i].name)

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
                        print(f"Saved Dataset {C.name}")
                        temp_df.to_csv(
                            f"{self.folders[2]}/{C.name}_reduced.csv", index=False)
                        C.pickle()

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
                    ax[0].set_ylabel('NEXAFS')
                    ax[0].tick_params(direction='in', labelsize=15, width=2)
                    ax[1].set_title(
                        'Background subtracted normalized & shifted')
                    ax[1].set_xlabel("Energy")
                    ax[1].set_ylabel('NEXAFS')
                    ax[1].yaxis.set_label_position("right")
                    ax[1].yaxis.tick_right()
                    ax[1].tick_params(direction='in', labelsize=15, width=2)

                    for i in range(len(ITN)):
                        ax[0].plot(energy[i][v1[i]:v2[i]], ITN[i])
                        ax[1].plot(energy[i][v1[i]:v2[i]], ITN[i] +
                                   0.1*(i+1), label=self.used_class_list[i].name)

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
                            print(f"Saved Dataset {C.name}")
                            temp_df.to_csv(
                                f"{self.folders[2]}/{C.name}_reduced.csv", index=False)
                            C.pickle()

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
            """Define a chebyshev polynomial using np.polynomial.chebyshev.fit method"""
            w = (1/y) ** n
            p = np.polynomial.Chebyshev.fit(x, y, d, w=w)

            return p(x)

        try:
            number = self.used_dataset_position
            df = self.used_df_type

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
            axs[0].set_ylabel('NEXAFS')
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
            axs[1].set_ylabel('NEXAFS')
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
                C = self.used_datasets
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
                print(f"Saved Dataset {C.name}")
                temp_df.to_csv(
                    f"{self.folders[2]}/{C.name}_reduced.csv", index=False)
                C.pickle()

            @ButtonRemoveBackground.on_click
            def ActionRemoveBackground(selfbutton):
                # Substract background to the intensity
                clear_output(True)

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                ax[0].set_title('Background subtracted')
                ax[0].set_xlabel("Energy")
                ax[0].set_ylabel('NEXAFS')
                ax[0].tick_params(direction='in', labelsize=15, width=2)
                ax[0].set_xlim(energy[number][v1[number]],
                               energy[number][v2[number]])

                ax[1].set_title('Background subtracted shifted')
                ax[1].set_xlabel("Energy")
                ax[1].set_ylabel('NEXAFS')
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
                                   0.1*(i), label=self.used_class_list[i].name)

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
                        print(f"Saved Dataset {C.name}")
                        temp_df.to_csv(
                            f"{self.folders[2]}/{C.name}_reduced.csv", index=False)
                        C.pickle()

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
                    ax[0].set_ylabel('NEXAFS')
                    ax[0].tick_params(direction='in', labelsize=15, width=2)
                    ax[0].set_xlim(energy[number][v1[number]],
                                   energy[number][v2[number]])
                    ax[1].set_title(
                        'Background subtracted normalized & shifted')
                    ax[1].set_xlabel("Energy")
                    ax[1].set_ylabel('NEXAFS')
                    ax[1].yaxis.set_label_position("right")
                    ax[1].yaxis.tick_right()
                    ax[1].tick_params(direction='in', labelsize=15, width=2)
                    ax[1].set_xlim(energy[number][v1[number]],
                                   energy[number][v2[number]])

                    for i in range(len(ITN)):
                        ax[0].plot(energy[i][v1[i]:v2[i]], ITN[i])
                        ax[1].plot(energy[i][v1[i]:v2[i]], ITN[i] + 0.1 *
                                   (i+1), label=self.used_class_list[i].name)

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
                            print(f"Saved Dataset {C.name}")
                            temp_df.to_csv(
                                f"{self.folders[2]}/{C.name}_reduced.csv", index=False)
                            C.pickle()

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
        """Reduce the background using a fixed number of points and Polynoms between them"""
        try:
            number = self.used_dataset_position
            df = self.used_df_type

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
                axs[0].set_ylabel('NEXAFS')
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
                axs[1].set_ylabel('NEXAFS')
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
                    C = self.used_datasets
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
                    print(f"Saved Dataset {C.name}")
                    C.pickle()
                    temp_df.to_csv(
                        f"{self.folders[2]}/{C.name}_reduced.csv", index=False)

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
                    ax[0].set_ylabel('NEXAFS')
                    ax[0].tick_params(direction='in', labelsize=15, width=2)
                    ax[0].set_xlim(energy[number][v1[number]],
                                   energy[number][v2[number]])

                    ax[1].set_title('Background subtracted shifted')
                    ax[1].set_xlabel("Energy")
                    ax[1].set_ylabel('NEXAFS')
                    ax[1].yaxis.tick_right()
                    ax[1].tick_params(direction='in', labelsize=15, width=2)
                    ax[1].set_xlim(energy[number][v1[number]],
                                   energy[number][v2[number]])
                    ax[1].yaxis.set_label_position("right")

                    for i in range(len(ITB)):
                        ax[0].plot(energy[i][v1[i]:v2[i]], ITB[i])
                        ax[1].plot(energy[i][v1[i]:v2[i]], ITB[i] +
                                   0.1 * (i), label=self.used_class_list[i].name)

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
                            print(f"Saved Dataset {C.name}")
                            C.pickle()
                            temp_df.to_csv(
                                f"{self.folders[2]}/{C.name}_reduced.csv", index=False)

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
                        ax[0].set_ylabel('NEXAFS')
                        ax[0].tick_params(
                            direction='in', labelsize=15, width=2)
                        ax[0].set_xlim(energy[number][v1[number]],
                                       energy[number][v2[number]])
                        ax[1].set_title(
                            'Background subtracted normalized & shifted')
                        ax[1].set_xlabel("Energy")
                        ax[1].set_ylabel('NEXAFS')
                        ax[1].yaxis.set_label_position("right")
                        ax[1].yaxis.tick_right()
                        ax[1].tick_params(
                            direction='in', labelsize=15, width=2)
                        ax[1].set_xlim(energy[number][v1[number]],
                                       energy[number][v2[number]])
                        for i in range(len(ITN)):
                            ax[0].plot(energy[i][v1[i]:v2[i]], ITN[i])
                            ax[1].plot(energy[i][v1[i]:v2[i]], ITN[i] + 0.1 *
                                       (i+1), label=self.used_class_list[i].name)

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
                                print(f"Saved Dataset {C.name}")
                                C.pickle()
                                temp_df.to_csv(
                                    f"{self.folders[2]}/{C.name}_reduced.csv", index=False)

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

    def reduce_single_spline(self, y, order, interval, cursor, param_A, param_B):
        """Single spline method to remove background
        """

        try:
            number = self.used_dataset_position
            df = self.used_df_type

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
            axs[0].set_ylabel('NEXAFS')
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
            axs[1].set_ylabel('NEXAFS')
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right()
            axs[1].set_xlim(energy[number][0], energy[number][-1])

            axs[1].plot(energy[number], normalized_data, '-', color='C0')

            @ButtonSaveDataset.on_click
            def ActionSaveDataset(selfbutton):
                # Save single Dataset without background in Class
                C = self.used_datasets
                temp_df = pd.DataFrame()
                temp_df["Energy"] = energy[number]
                temp_df["\u03BC"] = mu[number]
                temp_df["background_corrected"] = difference
                temp_df["\u03BC_variance"] = [
                    1 / d if d > 0 else 0 for d in difference]
                temp_df["second_normalized_\u03BC"] = normalized_data
                setattr(C, "reduced_df", temp_df)
                display(temp_df)
                print(f"Saved Dataset {C.name}")
                temp_df.to_csv(
                    f"{self.folders[2]}/{C.name}_reduced.csv", index=False)
                C.pickle()

            @ButtonRemoveBackground.on_click
            def ActionRemoveBackground(selfbutton):
                # Substract background to the intensity
                clear_output(True)

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                ax[0].set_title('Background subtracted')
                ax[0].set_xlabel("Energy")
                ax[0].set_ylabel('NEXAFS')

                ax[1].set_title('Background subtracted shifted')
                ax[1].set_xlabel("Energy")
                ax[1].set_ylabel('NEXAFS')
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
                                f"Parameters for {self.used_class_list[i].name}")
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
                                   label=self.used_class_list[i].name)

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
                        print(f"Saved Dataset {C.name}")
                        temp_df.to_csv(
                            f"{self.folders[2]}/{C.name}_reduced.csv", index=False)
                        C.pickle()

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
                    ax[0].set_ylabel('NEXAFS')
                    ax[0].tick_params(direction='in', labelsize=15, width=2)
                    ax[1].set_title(
                        'Background subtracted normalized & shifted')
                    ax[1].set_xlabel("Energy")
                    ax[1].set_ylabel('NEXAFS')
                    ax[1].yaxis.set_label_position("right")
                    ax[1].yaxis.tick_right()
                    ax[1].tick_params(direction='in', labelsize=15, width=2)

                    for i in range(len(ITN)):
                        ax[0].plot(energy[i], ITN[i])
                        ax[1].plot(energy[i], ITN[i]+0.1*(i+1),
                                   label=self.used_class_list[i].name)

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
                            print(f"Saved Dataset {C.name}")
                            temp_df.to_csv(
                                f"{self.folders[2]}/{C.name}_reduced.csv", index=False)
                            C.pickle()

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
            """Return the center point derivative for each point x_i as np.gradient(y) / np.gradient(x)"""
            dEnergy, dIT = [], []

            for i in range(len(mu)):
                x = energy[i].values
                y = mu[i].values

                dEnergy.append(x)
                dIT.append(np.gradient(y) / np.gradient(x))

            return dEnergy, dIT

        try:
            number = self.used_dataset_position
            df = self.used_df_type

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
                axs[0].set_ylabel('NEXAFS')
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
                axs[1].set_ylabel('NEXAFS')
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
                    setattr(self.used_datasets, "E0", dE[number][s])
                    self.used_datasets.pickle()
                    print(f"Saved E0 for {self.used_datasets.name};  ")

                def ActionSaveAll(selfbutton):
                    for j, C in enumerate(self.used_class_list):
                        setattr(
                            self.used_class_list[j], "E0", dE[j][maxima[j]])
                        C.pickle()
                        print(
                            f"Saved E0 for {self.used_class_list[j].name};  ")

                def ActionSplinesReduction(selfbutton):
                    try:
                        E0Values = [getattr(self.used_class_list[j], "E0")
                                    for j, C in enumerate(self.used_class_list)]

                        self._list_reduce_splines = interactive(
                            self.reduce_splines,
                            spec=widgets.Dropdown(
                                options=self.used_class_list,
                                description='Select the Dataset:',
                                disabled=False,
                                style={'description_width': 'initial'},
                                layout=Layout(width='60%')),
                            order_pre=widgets.Dropdown(
                                options=[("Select and order", "value"), ("Victoreen", "victoreen"), (
                                    "0", 0), ("1", 1), ("2", 2), ("3", 3)],
                                value="value",
                                description='Order of pre-edge:',
                                disabled=False,
                                style={'description_width': 'initial'}),
                            order_pst=widgets.Dropdown(
                                options=[("Select and order", "value"), ("Victoreen", "victoreen"), (
                                    "0", 0), ("1", 1), ("2", 2), ("3", 3)],
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

    def reduce_splines(self, spec, order_pre, order_pst, s1, s2, param_a1, param_b1, param_a2, param_b2, y):
        """Reduce the background using two curves and then normalize by edge-jump."""

        try:
            number = self.used_class_list.index(spec)
            df = self.used_df_type
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
            axs[0].set_ylabel('NEXAFS')
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
            axs[1].set_ylabel('NEXAFS')
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
                setattr(spec, "reduced_df_splines", temp_df)
                print(f"Saved Dataset {spec.name}")
                temp_df.to_csv(
                    f"{self.folders[2]}/{spec.name}_SplinesReduced.csv", index=False)

                # Need to plot again
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                axs[0].set_xlabel("Energy")
                axs[0].set_ylabel('NEXAFS')
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
                axs[1].set_ylabel('NEXAFS')
                axs[1].yaxis.set_label_position("right")
                axs[1].tick_params(direction='in', labelsize=15, width=2)
                axs[1].legend()
                plt.tight_layout()

                plt.savefig(
                    f"{self.folders[3]}/splines_reduced_{spec.name}.pdf")
                plt.savefig(
                    f"{self.folders[3]}/splines_reduced_{spec.name}.png")
                plt.close()
                spec.pickle()

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

        try:
            number = self.used_dataset_position
            df = self.used_df_type

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
            axs[0].set_ylabel('NEXAFS')
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
            axs[1].set_ylabel('NEXAFS')
            axs[1].yaxis.set_label_position("right")
            axs[1].yaxis.tick_right()
            axs[1].tick_params(direction='in', labelsize=15, width=2)
            axs[1].set_xlim(energy[number][v1[number]],
                            energy[number][v2[number]])

            axs[1].plot(energy[number][v1[number]:v2[number]], mu[number][v1[number]:v2[number]] / max(mu[number][v1[number]:v2[number]]), '-', color='C0')

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
                axs[0].set_ylabel('NEXAFS')
                axs[0].set_title('Data')

                axs[1].set_title('normalized data')
                axs[1].set_xlabel("Energy")
                axs[1].set_ylabel('NEXAFS')

                normalized_data = []
                for j, C in enumerate(self.used_class_list):
                    axs[0].plot(energy[j][v1[j]:v2[j]], mu[j]
                                [v1[j]:v2[j]], label=f"{C.name}")
                    axs[1].plot(energy[j][v1[j]:v2[j]], (mu[j][v1[j]:v2[j]] /
                                max(mu[j][v1[j]:v2[j]])), label=f"{C.name}")
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
                        print(f"Saved Dataset {C.name}")
                        temp_df.to_csv(
                            f"{self.folders[2]}/{C.name}_reduced.csv", index=False)
                        C.pickle()

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
    def define_fitting_df(self, spec, printed_df, show):
        """

        """

        if not show:
            print("Window cleared")
            clear_output(True)

        elif show:
            try:
                self.used_datasets = spec
                self.used_df_type = printed_df
                used_df = getattr(self.used_datasets, self.used_df_type)

                display(self.widget_list_define_model)

            except (AttributeError, KeyError):
                print(f"Wrong Dataset and column combination !")

    def define_model(
        self,
        xcol,
        ycol,
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
        """
        # Retrieve the data
        self.used_df_fit = getattr(self.used_datasets, self.used_df_type)
        clear_output(True)

        try:

            # We create the model
            try:
                i = int(np.where(self.used_df_fit[xcol] == interval[0])[0])
            except TypeError:
                i = 0

            try:
                j = int(np.where(self.used_df_fit[xcol] == interval[1])[0])
            except TypeError:
                j = -1

            y = self.used_df_fit[ycol].values[i:j]
            x = self.used_df_fit[xcol].values[i:j]

            self.fit_df = pd.DataFrame({
                xcol: x,
                ycol: y
            })

            # Background
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

            # Create a dictionnary for the peak to be able to iterate on their names
            peaks = dict()

            for i in range(peak_number):
                peaks[f"Peak_{i}"] = peak_type(prefix=f"P{i}_")
                self.pars.update(peaks[f"Peak_{i}"].make_params())
                self.mod += peaks[f"Peak_{i}"]

            self.parameter_names = [str(p) for p in self.pars]
            self.parameter_columns_names = ["value", "min", "max"]

            if fix_model:
                print(
                    "Please start by selecting a parameter to start working on the initial guess.")

                def InitPara(para, column, value):
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
                    display(widgets.HBox(
                        (ButtonRetrievePara, ButtonSavePara, ButtonGuess, ButtonFit, ButtonSaveModel)))
                    display(self.pars)

                    @ButtonRetrievePara.on_click
                    def ActionRetrievePara(selfbutton):
                        clear_output(True)
                        plt.close()
                        display(widgets.HBox(
                            (ButtonRetrievePara, ButtonSavePara, ButtonGuess, ButtonFit, ButtonSaveModel)))

                        try:
                            self.result = getattr(self.used_datasets, "result")
                            self.pars = getattr(self.result, "params")
                            print(
                                "Previously saved parameters loaded, press see current guess to see current guess.")

                        except:
                            print("Could not load any parameters.")

                    @ButtonSavePara.on_click
                    def ActionSavePara(selfbutton):
                        clear_output(True)
                        plt.close()
                        display(widgets.HBox(
                            (ButtonRetrievePara, ButtonSavePara, ButtonGuess, ButtonFit, ButtonSaveModel)))
                        try:
                            if column == "value":
                                self.pars[f"{para}"].set(value=value)
                            if column == "min":
                                self.pars[f"{para}"].set(min=value)
                            if column == "max":
                                self.pars[f"{para}"].set(max=value)

                            display(self.pars)
                        except Exception as e:
                            raise e

                    @ButtonGuess.on_click
                    def ActionGuess(selfbutton):
                        clear_output(True)
                        plt.close()
                        display(widgets.HBox(
                            (ButtonRetrievePara, ButtonSavePara, ButtonGuess, ButtonFit, ButtonSaveModel)))
                        try:
                            display(self.pars)

                            # Current guess
                            self.init = self.mod.eval(self.pars, x=x)

                            fig, ax = plt.subplots(figsize=(16, 6))
                            ax.plot(x, y, label="Data")
                            ax.plot(x, self.init, label='Current guess')
                            ax.legend()
                            plt.show()
                        except Exception as e:
                            raise e

                    @ButtonFit.on_click
                    def ActionFit(selfbutton):
                        clear_output(True)
                        display(widgets.HBox(
                            (ButtonRetrievePara, ButtonSavePara, ButtonGuess, ButtonFit, ButtonSaveModel)))

                        # Current guess
                        self.init = self.mod.eval(self.pars, x=x)

                        # Retrieve the interval
                        try:
                            i = int(
                                np.where(self.used_df_fit[xcol] == interval[0])[0])
                        except TypeError:
                            i = 0

                        try:
                            j = int(
                                np.where(self.used_df_fit[xcol] == interval[1])[0])
                        except TypeError:
                            j = -1

                        # Retrieve weights
                        if w == "Obs":
                            weights = 1/y.values[i:j]

                        elif w == "RMS":
                            try:
                                weights = 1/self.used_df_fit["RMS"].values[i:j]
                            except AttributeError:
                                print(
                                    "You need to define the RMS error first, the weights are put to 1 for now.")
                                weights = None

                        elif w == "user_error":
                            try:
                                weights = self.used_df_fit["user_error"].values[i:j]
                            except (KeyError, AttributeError):
                                print(
                                    "You need to define the User error in the initialisation, the weights are put to 1 for now.")
                                weights = None
                        else:
                            weights = None

                        self.fit_df["weights"] = weights

                        # Launch fit
                        self.out = self.mod.fit(
                            y, self.pars, x=x, method=method, weights=weights)
                        self.comps = self.out.eval_components(x=x)

                        display(self.out)

                        # Check the stats of the fit
                        chisq, p = chisquare(
                            self.out.data, self.out.best_fit, ddof=(self.out.nfree))
                        setattr(self.used_datasets, "chisq", chisq)
                        setattr(self.used_datasets, "p", p)

                        print(
                            f"Sum of squared residuals : {np.sum(self.out.residual**2)}, lmfit chisqr : {self.out.chisqr}")
                        print(
                            f"Sum of squared residuals/nfree : {np.sum(self.out.residual**2)/(self.out.nfree)}, lmfit redchisqr : {self.out.redchi}")

                        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
                        print(
                            f"Scipy Chi square for Poisson distri = {chisq}, 1 - p = {1 - p}")
                        print(
                            f"lmfit chisqr divided iter by expected : {np.sum((self.out.residual**2)/self.out.best_fit)}")

                        r_factor = 100 * \
                            (np.sum(self.out.residual**2)/np.sum(self.out.data**2))
                        setattr(self.used_datasets, "r_factor", r_factor)
                        print(f"R factor : {r_factor} %.\n")

                        # Plot
                        fig, axes = plt.subplots(2, 2, figsize=(
                            16, 7), gridspec_kw={'height_ratios': [5, 1]})

                        axes[0, 0].plot(x, y, label="Data")
                        axes[0, 0].plot(x, self.out.best_fit, label='Best fit')
                        axes[0, 0].set_xlabel(xcol, fontweight='bold')
                        axes[0, 0].set_ylabel(ycol, fontweight='bold')
                        axes[0, 0].set_title(
                            f"Best fit - {self.used_datasets.name}")
                        axes[0, 0].legend()

                        # Residuals
                        axes[1, 0].set_title("Residuals")
                        axes[1, 0].scatter(x, self.out.residual, s=0.5)
                        axes[1, 0].set_xlabel(xcol, fontweight='bold')
                        axes[1, 0].set_ylabel(ycol, fontweight='bold')

                        axes[1, 1].set_title("Residuals")
                        axes[1, 1].scatter(x, self.out.residual, s=0.5)
                        axes[1, 1].set_xlabel(xcol, fontweight='bold')
                        axes[1, 1].set_ylabel(ycol, fontweight='bold')

                        # Detailed plot
                        axes[0, 1].set_title("Best fit - Detailed")
                        axes[0, 1].plot(x, y, label="Data")

                        if background_type == ConstantModel:
                            axes[0, 1].plot(x, np.ones(
                                len(x)) * self.comps['Bcgd_'], 'k--', label='Background')
                        else:
                            axes[0, 1].plot(
                                x, self.comps['Bcgd_'], 'k--', label='Background')

                        if step_type:
                            axes[0, 1].plot(
                                x, self.comps['Step_'], label='Step')

                        for i in range(peak_number):
                            axes[0, 1].plot(
                                x, self.comps[f"P{i}_"], label=f"Peak nb {i}")

                        axes[0, 1].set_xlabel(xcol, fontweight='bold')
                        axes[0, 1].set_ylabel(ycol, fontweight='bold')
                        axes[0, 1].legend()
                        plt.tight_layout()

                        plt.savefig(
                            f"{self.folders[3]}/fit_{self.used_datasets.name}.pdf")
                        plt.savefig(
                            f"{self.folders[3]}/fit_{self.used_datasets.name}.png")

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
                                # Confidence interval with the standard error from the covariance matrix
                                print(
                                    f"The shape of the estimated covariance matrix is : {np.shape(self.out.covar)}. It is accessible under the self.out.covar attribute.")
                                self.ci = lmfit.conf_interval(
                                    self.out, self.out.result)
                                print(
                                    "The confidence intervals determined by the standard error from the covariance matrix are :")
                                lmfit.printfuncs.report_ci(self.ci)

                            except:
                                print("""
                                    No covariance matrix could be estimated from the fitting routine. 
                                    We determine the confidence intervals without standard error estimates, careful !
                                    Please refer to lmfit documentation for additional informations, 
                                    we set the standard error to 10 % of the parameter values.
                                    """)

                                # Determine confidence intervals without standard error estimates, careful !
                                for p in self.out.result.params:
                                    self.out.result.params[p].stderr = abs(
                                        self.out.result.params[p].value * 0.1)
                                self.ci = lmfit.conf_interval(
                                    self.out, self.out.result)
                                print(
                                    "The confidence intervals determined without standard error estimates are :")
                                lmfit.printfuncs.report_ci(self.ci)

                            try:
                                setattr(self.used_datasets,
                                        "confidence_intervals", self.ci)
                                self.used_datasets.pickle()
                            except Exception as E:
                                print("Confidence intervals could not be saved.\n")
                                raise E

                        @ButtonParaSpace.on_click
                        def ActionParaSpace(selfbutton):
                            return self.explore_params(i, j, xcol, ycol)

                    @ButtonSaveModel.on_click
                    def ActionSave(selfbutton):
                        clear_output(True)
                        display(widgets.HBox(
                            (ButtonRetrievePara, ButtonSavePara, ButtonGuess, ButtonFit, ButtonSaveModel)))

                        print("Saved the initial parameters as Interface.pars")
                        self.pars = self.out.params

                        try:
                            setattr(self.used_datasets, "init", self.init)
                            print("Saved the initial guess as Dataset.init")

                            setattr(self.used_datasets,
                                    "result", self.out.result)
                            print(
                                "Saved the output of the fitting routine as Dataset.result ")
                        except:
                            print("Launch the fit first. \n")

                        try:
                            self.fit_df["fit"] = self.out.best_fit
                            self.fit_df["residuals"] = self.out.residual
                            setattr(self.used_datasets, "fit_df", self.fit_df)
                        except Exception as e:
                            raise e

                        try:
                            self.used_datasets.pickle()

                        except Exception as e:
                            print("Could not save the class instance with pickle().")
                            raise e

                self._list_parameters_fit = interactive(
                    InitPara,
                    para=widgets.Dropdown(
                        options=self.parameter_names,
                        value=None,
                        description='Select the parameter:',
                        style={'description_width': 'initial'}),
                    column=widgets.Dropdown(
                        options=self.parameter_columns_names,
                        description='Select the column:',
                        style={'description_width': 'initial'}),
                    value=widgets.FloatText(
                        value=0,
                        step=0.01,
                        description='Value :'))
                self.widget_list_parameters_fit = widgets.VBox([widgets.HBox(
                    self._list_parameters_fit.children[0:3]), self._list_parameters_fit.children[-1]])
                display(self.widget_list_parameters_fit)

            else:
                plt.close()
                print("Cleared")
                clear_output(True)

        except (AttributeError, KeyError):
            plt.close()
            if ycol == "value":
                print("Please select a column.")
            else:
                print(f"Wrong Dataset and column combination !")
        except TypeError:
            print("This peak distribution is not yet working, sorry.")

    def explore_params(self, i, j, xcol, ycol):
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

            y = self.used_df_fit[ycol].values[i:j]
            x = self.used_df_fit[xcol].values[i:j]

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

            self.emcee_plot = corner.corner(self.resi.flatchain, labels=self.resi.var_names, truths=list(
                self.resi.params.valuesdict().values()))
            plt.savefig(
                f"{self.folders[3]}/{self.used_datasets.name}_corner_plot.pdf")
            plt.savefig(
                f"{self.folders[3]}/{self.used_datasets.name}_corner_plot.png")

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
                        name, param.value, self.resi.params[name].value, self.resi.params[name].stderr))

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
            if ycol == "value":
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
        if check_plot == "Zero":
            print("No plotting.")

        elif check_plot == "Plot" and len(spec_number) == 1:
            @interact(
                interval=widgets.FloatRangeSlider(
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
                    layout=Layout(width="50%", height='40px')),
                colorplot=widgets.ColorPicker(
                    concise=False,
                    description='Pick a color',
                    value='Blue',
                    disabled=False,
                    style={'description_width': 'initial'}))
            def plot_one(interval, colorplot):
                try:
                    used_df = getattr(spec_number[0], plot_df)
                    try:
                        v1 = int(np.where(used_df[x] == interval[0])[0])
                    except TypeError:
                        v1 = 0

                    try:
                        v2 = int(np.where(used_df[x] == interval[1])[0])
                    except TypeError:
                        v2 = -1

                    ButtonSavePlot = Button(
                        description="Save Plot",
                        layout=Layout(width='15%', height='35px'))
                    display(ButtonSavePlot)

                    plt.close()

                    fig, ax = plt.subplots(figsize=(16, 6))
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    ax.set_title(title)

                    ax.plot(used_df[x][v1:v2], used_df[y][v1:v2], linewidth=1,
                            color=colorplot, label=f"{spec_number[0].name}")
                    ax.legend()

                    @ButtonSavePlot.on_click
                    def ActionSavePlot(selfbutton):
                        fig, ax = plt.subplots(figsize=(16, 6))
                        ax.set_xlabel(x_axis)
                        ax.set_ylabel(y_axis)
                        ax.set_title(title)

                        ax.plot(used_df[x][v1:v2], used_df[y][v1:v2], linewidth=1,
                                color=colorplot, label=f"{spec_number[0].name}")
                        ax.legend()
                        plt.tight_layout()
                        plt.savefig(f"{self.folders[3]}/{title}.pdf")
                        plt.savefig(f"{self.folders[3]}/{title}.png")
                        print(f"Figure {title} saved !")
                        plt.close()

                except AttributeError:
                    plt.close()
                    print(
                        f"This class does not have the {plot_df} dataframe associated yet.")
                except IndexError:
                    plt.close()
                    print(f"Please select at least one spectra.")
                except KeyError:
                    plt.close()
                    print(
                        f"The {plot_df} dataframe does not have such attributes.")

        elif check_plot == "Plot" and len(spec_number) > 1:
            try:
                T = [int(C.logbook_entry["Temp (K)"]) for C in spec_number]
                print("The color is function of the temperature for each Dataset.")
            except:
                print(
                    "No valid logbook entry for the temperature found as [Temp (K)], the color of the plots will be random.")
                T = False

            @interact(interval=widgets.FloatRangeSlider(
                min=self.new_energy_column[0],
                value=[self.new_energy_column[0], self.new_energy_column[-1]],
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
            def plot_all(interval):
                try:
                    plt.close()
                    fig, ax = plt.subplots(figsize=(16, 6))
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    ax.set_title(title)

                    for j, C in enumerate(spec_number):
                        used_df = getattr(C, plot_df)
                        try:
                            v1 = int(np.where(used_df[x] == interval[0])[0])
                        except TypeError:
                            v1 = 0

                        try:
                            v2 = int(np.where(used_df[x] == interval[1])[0])
                        except TypeError:
                            v2 = -1

                        if T:
                            ax.plot(used_df[x][v1: v2], used_df[y][v1: v2], linewidth=1, label=f"{spec_number[j].name}", color=(
                                (T[j]-273.15)/(max(T)-273.15), 0, ((max(T)-273.15)-(T[j]-273.15))/(max(T)-273.15)))

                        if not T:
                            ax.plot(used_df[x][v1: v2], used_df[y][v1: v2],
                                    linewidth=1, label=f"{spec_number[j].name}")
                    ax.legend(loc='upper center', bbox_to_anchor=(
                        0.5, -0.2), fancybox=True, shadow=True, ncol=5)

                    ButtonSavePlot = Button(
                        description="Save Plot",
                        layout=Layout(width='15%', height='35px'))
                    display(ButtonSavePlot)

                    @ButtonSavePlot.on_click
                    def ActionSavePlot(selfbutton):
                        plt.close()
                        fig, ax = plt.subplots(figsize=(16, 6))
                        ax.set_xlabel(x_axis)
                        ax.set_ylabel(y_axis)
                        ax.set_title(title)

                        for j, C in enumerate(spec_number):
                            used_df = getattr(C, plot_df)
                            try:
                                v1 = int(
                                    np.where(used_df[x] == interval[0])[0])
                            except TypeError:
                                v1 = 0

                            try:
                                v2 = int(
                                    np.where(used_df[x] == interval[1])[0])
                            except TypeError:
                                v2 = -1
                            if T:
                                ax.plot(used_df[x][v1: v2], used_df[y][v1: v2], linewidth=1, label=f"{spec_number[j].name}", color=(
                                    (T[j]-273.15)/(max(T)-273.15), 0, ((max(T)-273.15)-(T[j]-273.15))/(max(T)-273.15)))

                            if not T:
                                ax.plot(used_df[x][v1: v2], used_df[y][v1: v2],
                                        linewidth=1, label=f"{spec_number[j].name}")

                        ax.legend(loc='upper center', bbox_to_anchor=(
                            0.5, -0.2), fancybox=True, shadow=True, ncol=5)
                        plt.tight_layout()
                        plt.savefig(f"{self.folders[3]}/{title}.pdf")
                        plt.savefig(f"{self.folders[3]}/{title}.png")
                        print(f"Figure {title} saved !")
                        plt.close()

                except AttributeError:
                    plt.close()
                    print(
                        f"This class does not have the {plot_df} dataframe associated yet.")
                except KeyError:
                    plt.close()
                    print(
                        f"The {plot_df} dataframe does not have such attributes.")

        elif check_plot == "3D" and len(spec_number) > 1:
            print("Please pick a valid range for the x axis.")
            # try:
            #     T = [int(C.logbook_entry["Temp (K)"]) for C in spec_number]
            #     print("The color is function of the temperature for each Dataset.")
            # except:
            #     print("No valid logbook entry for the temperature found as [Temp (K)], the color of the plots will be random.")
            #     T = False

            try:
                # Create a df that spans the entire energy range
                self.merged_values = pd.DataFrame({
                    x: self.new_energy_column
                })

                for C in spec_number:
                    used_df = getattr(C, plot_df)
                    yvalues = pd.DataFrame(
                        {x: used_df[x].values, y: used_df[y].values})

                    for v in self.merged_values[x].values:
                        if v not in yvalues[x].values:
                            yvalues = yvalues.append({x: v}, ignore_index=True).sort_values(
                                by=[x]).reset_index(drop=True)

                    self.merged_values[str(C.name) + "_"+str(y)] = yvalues[y]

                def three_d_plot(xname, yname, zname, dist, elev, azim, cmap_style, title, interval):
                    try:
                        # Get the data
                        clear_output(True)
                        v1 = int(
                            np.where(self.new_energy_column == interval[0])[0])
                        v2 = int(
                            np.where(self.new_energy_column == interval[1])[0])

                        data = self.merged_values.copy()[v1:v2]
                        data.index = data["Energy"]
                        del data["Energy"]

                        display(data)

                        df = data.unstack().reset_index()
                        df.columns = ["X", "Y", "Z"]

                        df['X'] = pd.Categorical(df['X'])
                        df['X'] = df['X'].cat.codes

                        NonNanValues = [j for j in df["Z"] if not np.isnan(j)]

                        # Make the plot
                        fig = plt.figure(figsize=(15, 10))
                        ax = fig.add_subplot(111, projection='3d')
                        ax.dist = dist
                        ax.elev = elev
                        ax.azim = azim

                        # Add a color bar which maps values to colors. viridis or jet
                        surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=cmap_style, linewidth=0.2, vmin=min(
                            NonNanValues), vmax=max(NonNanValues))
                        colorbarplot = fig.colorbar(
                            surf, shrink=0.6, label="Colorbar")

                        ax.set_xlabel(xname)
                        ax.set_ylabel(yname)
                        ax.set_zlabel(zname)
                        ax.set_title(title)

                        fig.tight_layout()
                        plt.savefig(f"{self.folders[3]}/{title}.pdf")
                        plt.savefig(f"{self.folders[3]}/{title}.png")
                        plt.show()

                    except IndexError:
                        print("Please pick a valid range for the x axis.")

                _list_3D = interactive(
                    three_d_plot,
                    xname=widgets.Text(
                        value="Energy",
                        placeholder="x_axis",
                        description='Name of x axis:',
                        disabled=False,
                        continuous_update=False,
                        style={'description_width': 'initial'}),
                    yname=widgets.Text(
                        value="Temperature",
                        placeholder="y_axis",
                        description='Name of y axis:',
                        disabled=False,
                        continuous_update=False,
                        style={'description_width': 'initial'}),
                    zname=widgets.Text(
                        value="normalized EXAFS intensity",
                        placeholder="zaxis",
                        description='Name of z axis:',
                        disabled=False,
                        continuous_update=False,
                        style={'description_width': 'initial'}),
                    title=widgets.Text(
                        value="Evolution of edge with temperature",
                        placeholder="3D plot",
                        description='title:',
                        disabled=False,
                        continuous_update=False,
                        style={'description_width': 'initial'}),
                    dist=widgets.IntSlider(
                        value=10,
                        min=0,
                        max=50,
                        step=1,
                        description='Distance:',
                        disabled=False,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format="d",
                        style={'description_width': 'initial'}),
                    elev=widgets.IntSlider(
                        value=45,
                        min=0,
                        max=90,
                        step=1,
                        description='Elevation:',
                        disabled=False,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format="d",
                        style={'description_width': 'initial'}),
                    azim=widgets.IntSlider(
                        value=285,
                        min=0,
                        max=360,
                        step=1,
                        description='Azimuthal:',
                        disabled=False,
                        continuous_update=False,
                        orientation="horizontal",
                        readout=True,
                        readout_format="d",
                        style={'description_width': 'initial'}),
                    cmap_style=widgets.Dropdown(
                        options=[("Viridis", plt.cm.viridis), ("Jet", plt.cm.jet), ("Plasma", plt.cm.plasma), (
                            "Cividis", plt.cm.cividis), ("Magma", plt.cm.magma), ("Inferno", plt.cm.inferno)],
                        description='Select the style:',
                        continuous_update=False,
                        disabled=False,
                        style={'description_width': 'initial'}),
                    interval=widgets.FloatRangeSlider(
                        min=self.new_energy_column[0],
                        value=[self.new_energy_column[-2],
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
                        layout=Layout(width="50%", height='40px')))
                widget_list_3D = widgets.VBox([widgets.HBox(_list_3D.children[:3]), widgets.HBox(
                    _list_3D.children[3:6]), widgets.HBox(_list_3D.children[6:9]), _list_3D.children[-1]])
                display(widget_list_3D)

            except (AttributeError, KeyError):
                plt.close()
                print(f"Wrong Dataset and column combination !")
            except PermissionError:
                plt.close()
                print(f"Figure with same name opened in another program.")

        elif len(spec_number) == 0:
            print("Select more datasets.")

        elif check_plot == "3D" and len(spec_number) < 2:
            print("Select more datasets.")

    # Logbook interactive function
    def print_logbook(self, logbook_name, logbook_bool, column, value):
        """Allows one to filter multiple logbook columns by specific values
        """

        # Work on filtering values
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False

        try:
            value = int(value)
        except ValueError:
            value = value

        # Show logbook
        if not logbook_bool:
            global this_logbook
            try:
                logbook = pd.read_excel(logbook_name)
                ButtonFilterlogbook = Button(
                    description="Add a filter",
                    layout=Layout(width='15%', height='35px'))
                ButtonAssociatelogbook = Button(
                    description="Associate logbook entry to classes",
                    layout=Layout(width='25%', height='35px'))
                display(widgets.HBox(
                    (ButtonFilterlogbook, ButtonAssociatelogbook)))

                @ButtonFilterlogbook.on_click
                def ActionFilterlogbook(selfbutton):
                    clear_output(True)
                    global this_logbook
                    try:
                        # determine mask as series
                        mask = this_logbook[column] == value
                        this_logbook = this_logbook[mask]     # apply mask
                        display(this_logbook)
                    except NameError:
                        try:
                            # so that each consequent mask is still here,
                            # here we apply the first mask and create logbook
                            this_logbook = logbook
                            mask = this_logbook[column] == value
                            this_logbook = this_logbook[mask]
                            display(this_logbook)
                        except:
                            print("Wrong logbook name")
                    except KeyError:
                        print("Wrong column value")

                @ButtonAssociatelogbook.on_click
                def ActionAssociatelogbook(selfbutton):
                    logbook = pd.read_excel(logbook_name)

                    for items in logbook.iterrows():
                        if "scan_" in items[1].name:
                            namelog = items[1].name.split("scan_")[1]

                            for C in self.class_list:
                                nameclass = C.name.split("Dataset_")[
                                    1].split("~")[0]
                                if namelog == nameclass:
                                    try:
                                        setattr(C, "logbook_entry", items[1])
                                        print(
                                            f"The logbook has been correctly associated for {C.name}.")
                                        C.pickle()
                                    except:
                                        print(
                                            f"The logbook has been correctly associated for {C.name}, but could not be pickled, i.e. it will not be saved after this working session.\n")

                        else:
                            namelog = items[1].name

                            for C in self.class_list:
                                nameclass = C.name.split("Dataset_")[
                                    1].split("~")[0]
                                if namelog == nameclass:
                                    try:
                                        setattr(C, "logbook_entry", items[1])
                                        print(
                                            f"The logbook has been correctly associated for {C.name}.")
                                        C.pickle()
                                    except:
                                        print(
                                            f"The logbook has been correctly associated for {C.name}, but could not be pickled, i.e. it will not be saved after this working session.\n")

                                else:
                                    print(
                                        "The logbook entries and datasets could not be associated. Please refer to readme for guidelines on how to name your data.")

                            # except ValueError:
                            #     print("""This routine assumes that the name of each Dataset is stored in a \"Name\" column.\n
                            #         The names can be the same name as the datasets given in entry to the program,\n
                            #         In this case, the names must be possible to convert to intergers.\n
                            #         Otherwise, the names can be preceded of \"scan_\", followed by the Dataset number.\n
                            #         E.g, we have a file \"215215.txt\" in entry, and in the logbook, its name is either 215215 or scan_215215.""")

            except Exception as e:
                print("logbook not available.")
                raise e

            try:
                display(this_logbook)
            except NameError:
                try:
                    display(logbook)
                except Exception as e:
                    print("Wrong name")
                    raise e

        else:
            try:
                del this_logbook
            except:
                pass
            print("The logbook has been reset.")
            clear_output(True)

    # All handler functions

    def name_handler(self, change):
        """
        Handles changes on the widget used for the definition of the
        data_folder's name
        """
        if change.new:
            self._list_widgets_init.children[0].disabled = True
            self._list_widgets_init.children[2].disabled = False

        elif not change.new:
            self._list_widgets_init.children[0].disabled = False
            self._list_widgets_init.children[2].disabled = True

    def create_handler(self, change):
        """Handles changes on the widget used for the creation of subfolders"""

        if change.new:
            for w in self._list_widgets_init.children[3:7]:
                w.disabled = False
            self._list_widgets_init.children[1].disabled = True
            self._list_widgets_init.children[9].disabled = False
            self._list_widgets_init.children[10].disabled = False

        elif not change.new:
            for w in self._list_widgets_init.children[3:7]:
                w.disabled = True
            self._list_widgets_init.children[1].disabled = False
            self._list_widgets_init.children[9].disabled = True
            self._list_widgets_init.children[10].disabled = True

    def excel_handler(self, change):
        if change.new not in [".xlsx", ".nxs"]:
            for w in self._list_widgets_init.children[4:7]:
                w.disabled = False
        if change.new in [".xlsx", ".nxs"]:
            for w in self._list_widgets_init.children[4:7]:
                w.disabled = True

    def marker_handler(self, change):
        if change.new:
            for w in self._list_widgets_init.children[7:9]:
                w.disabled = False
            self._list_widgets_init.children[3].disabled = True
        if not change.new:
            for w in self._list_widgets_init.children[7:9]:
                w.disabled = True
            self._list_widgets_init.children[3].disabled = False

    def delete_handler(self, change):
        """
        Handles changes on the widget used for the deletion of previous work
        """

        if not change.new:
            self._list_widgets_init.children[10].disabled = False
        elif change.new:
            self._list_widgets_init.children[10].disabled = True

    def work_handler(self, change):
        """
        Handles changes on the widget used for marking the beginning of data
        treatment
        """
        if change.new:
            for w in self._list_widgets_init.children[:10]:
                w.disabled = True
        elif not change.new:
            for w in self._list_widgets_init.children[1:7]:
                w.disabled = False
            self._list_widgets_init.children[9].disabled = False

    def show_data_handler(self, change):
        """
        Handles changes on the widget used to decide whether or not we start
        the reduction in the visualization tab.
        """
        if change.new:
            self._list_data.children[0].disabled = True
            self._list_data.children[1].disabled = True
        elif not change.new:
            self._list_data.children[0].disabled = False
            self._list_data.children[1].disabled = False

    def relative_shift_bool_handler(self, change):
        if change.new:
            for w in self._list_relative_shift.children[:4]:
                w.disabled = True
        elif not change.new:
            for w in self._list_relative_shift.children[:4]:
                w.disabled = False

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

    def merge_bool_handler(self, change):
        """
        Handles changes on the widget used to decide whether or not we merge
        the energies in the tools tab.
        """

        if change.new:
            for w in self._list_merge_energies.children[0:5]:
                w.disabled = True
        elif not change.new:
            for w in self._list_merge_energies.children[0:5]:
                w.disabled = False

    def error_extraction_handler(self, change):
        """
        Handles changes on the widget used to decide whether or not we merge
        the energies in the tools tab.
        """

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
