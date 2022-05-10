import numpy as np
import pandas as pd
import tables as tb
from datetime import datetime
import glob
import os
import pickle
import h5py

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import display, Markdown, Latex, clear_output

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Legend, RangeTool, HoverTool, WheelZoomTool, CrosshairTool, LabelSet, LegendItem
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, export_png
from bokeh.palettes import gray, inferno, viridis, magma, cividis
output_notebook()


class DiamondDataset:
    """
    Designed for data analysis of XPS measurements at B07 at Diamond.
    """

    def __init__(self, filename):
        """
        DataFrame intensity is spectrum / iterations

        :param filename: path to NeXuS file
        """
        self.filename = filename.split("/")[-1]
        self.df_names = []

        # Extract all interesting attributes from file
        with h5py.File(self.filename) as f:
            print(self.filename)

            self.group_list = list(f["entry1"]["instrument"].keys())
            self.sequences = self.group_list[:-4]
            print("\tSequences recorded:", self.sequences)

            # Create first lists
            self.Concatenated_count_time = []
            self.Concatenated_iterations = []
            self.Concatenated_step_energy = []
            self.Concatenated_pass_energy = []
            self.Concatenated_photon_energy = []
            self.Concatenated_work_function = []
            # self.Concatenated_y_scale = []

            # Iterate over sequences
            for seq in self.sequences:
                self.group = seq

                # Save data
                setattr(self, f"{seq}_binding_energy",
                        f["entry1"]["instrument"][self.group]["binding_energy"][0, :])
                setattr(self, f"{seq}_kinetic_energy",
                        f["entry1"]["instrument"][self.group]["kinetic_energy"][0, :])
                setattr(self, f"{seq}_images",
                        f["entry1"]["instrument"][self.group]["images"][:])
                setattr(self, f"{seq}_spectra",
                        f["entry1"]["instrument"][self.group]["spectra"][:])
                setattr(self, f"{seq}_spectrum",
                        f["entry1"]["instrument"][self.group]["spectrum"][0, :])

                # Save metadata
                setattr(self, f"{seq}_count_time",
                        f["entry1"]["instrument"][self.group]["count_time"][0])
                setattr(self, f"{seq}_iterations",
                        f["entry1"]["instrument"][self.group]["iterations"][0])
                setattr(self, f"{seq}_step_energy",
                        f["entry1"]["instrument"][self.group]["step_energy"][0])
                setattr(self, f"{seq}_pass_energy",
                        f["entry1"]["instrument"][self.group]["pass_energy"][0])
                setattr(self, f"{seq}_photon_energy",
                        f["entry1"]["instrument"][self.group]["photon_energy"][0])
                setattr(self, f"{seq}_work_function",
                        f["entry1"]["instrument"][self.group]["work_function"][0])
                # setattr(self, f"{seq}_y_scale",
                #         f["entry1"]["instrument"][self.group]["y_scale"][:])
                setattr(self, f"{seq}_norm_y", 1)  # No normalisation for now
                setattr(self, f"{seq}_norm_range", (0, -1))  # No normalisation

                self.Concatenated_count_time.append(
                    getattr(self, f"{seq}_count_time"))
                self.Concatenated_iterations.append(
                    getattr(self, f"{seq}_iterations"))
                self.Concatenated_step_energy.append(
                    getattr(self, f"{seq}_step_energy"))
                self.Concatenated_pass_energy.append(
                    getattr(self, f"{seq}_pass_energy"))
                self.Concatenated_photon_energy.append(
                    getattr(self, f"{seq}_photon_energy"))
                self.Concatenated_work_function.append(
                    getattr(self, f"{seq}_work_function"))
                # self.Concatenated_y_scale.append(f"{seq}_y_scale")

                try:
                    # Save df
                    df = pd.DataFrame({
                        "binding_energy": getattr(self, f"{seq}_binding_energy"),
                        "kinetic_energy": getattr(self, f"{seq}_kinetic_energy"),
                        "intensity": getattr(self, f"{seq}_spectrum") / getattr(self, f"{seq}_iterations"),
                    })

                    setattr(self, f"{seq}_df", df)
                    self.df_names.append(f"{seq}_df")

                except Exception as E:
                    raise E

        # Create Concatenated df with all the df together
        try:
            if len(self.df_names) == 1:
                self.Concatenated_df = getattr(self, self.df_names[0]).copy()

            else:
                for j, df_name in enumerate(self.df_names):
                    df = getattr(self, df_name)
                    if j == 0:
                        binding_energy = df.binding_energy.values
                        kinetic_energy = df.kinetic_energy.values
                        intensity = df.intensity.values
                    else:
                        binding_energy = np.concatenate(
                            (binding_energy, df.binding_energy.values))
                        kinetic_energy = np.concatenate(
                            (kinetic_energy, df.kinetic_energy.values))
                        intensity = np.concatenate(
                            (intensity, df.intensity.values))

                self.Concatenated_df = pd.DataFrame({
                    "binding_energy": binding_energy,
                    "kinetic_energy": kinetic_energy,
                    "intensity": intensity,
                })

            self.Concatenated_count_time = np.mean(
                self.Concatenated_count_time)
            self.Concatenated_iterations = np.mean(
                self.Concatenated_iterations)
            self.Concatenated_step_energy = np.mean(
                self.Concatenated_step_energy)
            self.Concatenated_pass_energy = np.mean(
                self.Concatenated_pass_energy)
            self.Concatenated_photon_energy = np.mean(
                self.Concatenated_photon_energy)
            self.Concatenated_work_function = np.mean(
                self.Concatenated_work_function)
            # self.Concatenated_y_scale = np.mean(self.Concatenated_y_scale)

            # Sort values
            self.Concatenated_df.sort_values(
                by="binding_energy", ascending=False,
                inplace=True, ignore_index=True)

            self.df_names.append("Concatenated_df")

            print("\tAppended Concatenated_df.")
        except UnboundLocalError:
            print("\tEmpty file.")

    def __repr__(self):
        return self.filename

    def __str__(self):
        return repr(self)

    def plot_iterations(
        self,
        used_df,
        intensity_shift=0,
        norm_range=[0, 1000],
        title="Iterations",
        cmap="viridis",
        cmap_shift=0,
        x_axis="binding_energy",
        label_start=None,
        label_end=None,
    ):
        """
        Plot each iteration in spectra attribute
        :param used_df: options are listed in self.df_names
        :param cmap: options are gray, inferno, viridis, magma, cividis
        :param cmap_shift: used to avoid yellow
        """

        # Get data
        if used_df in self.df_names:
            data = getattr(self, used_df[:-3]+"_spectra")
            x = getattr(self, used_df[:-3]+"_" + x_axis)
            nb_iterations = data.shape[1]
        else:
            raise AttributeError(
                f"Please pick a DataFrame name from {self.df_names}")

        # Get color map
        cmap_dict = {
            "gray": gray,
            "inferno": inferno,
            "viridis": viridis,
            "magma": magma,
            "cividis": cividis
        }
        try:
            colors = cmap_dict[cmap](nb_iterations+cmap_shift)
        except Exception as e:
            raise e

        # Sort normalisation range
        norm_range = [np.min(norm_range), np.max(norm_range)]

        # Create figure
        TOOLTIPS = [
            (f"Binding energy (eV)", "$x"),
            ("Intensity", "$y"),
        ]

        p = figure(
            tools="xpan, pan, wheel_zoom, box_zoom, reset, undo, redo, crosshair, hover, save",
            active_scroll="wheel_zoom",
            x_axis_label="Binding energy",
            title=title,
            width=1100,
            height=600,
        )

        p.xaxis.axis_label_text_font_size = "15pt"
        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"
        p.title.text_font_size = '20pt'
        p.x_range.flipped = True if x_axis == "binding_energy" else False

        p.add_layout(Legend(label_text_font_size="15pt"), "center")

        for j in range(nb_iterations):
            # Normalize data
            y = data[:, j, :][0]
            y_norm = y / np.mean(data[:, j, (x < norm_range[1])
                                 & (x > norm_range[0])][0])

            # Add shift here if needed
            source = ColumnDataSource(
                data=dict(
                    x=x,
                    y=y_norm+j*intensity_shift,
                ))

            # Assign color
            color = colors[j]

            if j == 0 and label_start != None:
                p.line(
                    x='x', y='y', source=source,
                    line_width=1.2,
                    line_color=color,
                    hover_line_alpha=1.0,
                    hover_line_width=2.0,
                    legend_label=label_start,
                )

            elif j == nb_iterations-1 and label_end != None:
                p.line(
                    x='x', y='y', source=source,
                    line_width=1.2,
                    line_color=color,
                    hover_line_alpha=1.0,
                    hover_line_width=2.0,
                    legend_label=label_end,
                )
            else:
                p.line(
                    x='x', y='y', source=source,
                    line_width=1.2,
                    line_color=color,
                    hover_line_alpha=1.0,
                    hover_line_width=2.0,
                )

        show(p)

    def plot_data(
        self,
        df_name,
        x_axis="binding_energy",
        y_axis="spectrum"
    ):
        """
        Interactive plotting using Bokeh to allow the use of the cursor and zoom

        :param df_name: name of df, chosen in
        :param x_axis: "binding_energy" or "kinetic_energy"
        :param y_axis: "usually spectrum"
        """
        TOOLTIPS = [
            (f"Binding energy (eV)", "$x"),
            ("Intensity", "$y"),
        ]
        try:
            x = getattr(self, df_name[:-3]+"_"+x_axis)
            y = getattr(self, df_name[:-3]+"_"+y_axis)
        except AttributeError:
            print("Choose df name in", self.df_names)

        p = figure(
            tools="xpan, pan, wheel_zoom, box_zoom, reset, undo, redo, crosshair, hover, save",
            active_scroll="wheel_zoom",
            x_axis_label="Binding energy",
            title=df_name,
            width=1100,
            height=600,
        )

        p.xaxis.axis_label_text_font_size = "15pt"
        p.xaxis.major_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        p.yaxis.major_label_text_font_size = "15pt"
        p.title.text_font_size = '20pt'
        p.x_range.flipped = True
        p.add_layout(Legend(label_text_font_size="15pt"), "center")

        source = ColumnDataSource(
            data=dict(
                x=x,
                y=y,
            ))

        p.line(
            x='x', y='y', source=source,
            line_width=1.2,
            hover_line_alpha=1.0,
            hover_line_width=2.0,
        )

        show(p)

    def pickle_dataset(self, path):
        """Use the pickle module to save the classes"""
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        except PermissionError:
            print("""
                Permission denied, You cannot save this file because you \
                are not its creator. The changes are updated for this session \
                and you can still plot figures but once you exit the program, \
                all changes will be erased.
                """)

    @staticmethod
    def unpickle(path):
        """
        Use the pickle module to load the classes
        :param file_path: absolute path to pickle file
        """

        with open(path, 'rb') as f:
            return pickle.load(f)
