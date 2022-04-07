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


class DiamondDataset():
    """
    """

    def __init__(self, filename):
        """"""
        self.filename = filename.split("/")[-1]
        self.df_names = []

        with h5py.File(self.filename) as f:
            print(self.filename)

            self.group_list = list(f["entry1"]["instrument"].keys())
            self.sequences = self.group_list[:-4]
            print("\tSequences recorded:", self.sequences)
            for seq in self.sequences:
                self.group = seq

                # Save data
                setattr(self, f"{seq}_binding_energy",
                        f["entry1"]["instrument"][self.group]["binding_energy"][0, :])
                setattr(self, f"{seq}_kinetic_energy",
                        f["entry1"]["instrument"][self.group]["kinetic_energy"][0, :])
                setattr(self, f"{seq}_images",
                        f["entry1"]["instrument"][self.group]["images"][:])
                setattr(
                    self, f"{seq}_spectra", f["entry1"]["instrument"][self.group]["spectra"][:])
                setattr(
                    self, f"{seq}_spectrum", f["entry1"]["instrument"][self.group]["spectrum"][0, :])

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
                setattr(
                    self, f"{seq}_y_scale", f["entry1"]["instrument"][self.group]["y_scale"][:])

                try:
                    # Create array
                    arr = np.array([
                        getattr(self, f"{seq}_binding_energy"),
                        getattr(self, f"{seq}_kinetic_energy"),
                        getattr(self, f"{seq}_spectrum"),
                    ])

                    # Save df
                    df = pd.DataFrame(
                        arr.T, columns=["binding_energy", "kinetic_energy", "intensity"])
                    setattr(self, f"{seq}_df", df)
                    self.df_names.append(f"{seq}_df")

                except Exception as E:
                    raise E

    def __repr__(self):
        return self.filename
