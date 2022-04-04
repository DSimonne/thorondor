import numpy as np
import pandas as pd
import tables as tb
from datetime import dateti
import glob
import os
import pickle

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import display, Markdown, Latex, clear_output


class Dataset():
    """
    A new instance of the Dataset class will be initialized for each Dataset
    saved in the data folder. This object is then modified by the class gui
    For each Dataset, all the different dataframes that will be created as well
    as specific information e.g. E0 the edge jump can be find as attributes
    of this class. Certain attributes are instanced directly with the class,
    such as :
        _ Name of Dataset
        _ Path of original Dataset
        _ timestamp

    At the end of the data reduction, each Dataset should have at least three
    different data sets as attributes, saved as pandas.DataFrame:
        _ df : Original data
        _ shifted_df : Is one shifts the energy
        _ reduced_df : If one applies some background reduction or
           normalization method
        _ reduced_df_splines : If one applied the specific Splines background
           reduction and normalization method.

    The attributes of the class can be saved as an hdf5 file as well by using
    the Dataset.to_hdf5 fucntion.
    The pandas.DataFrame.to_hdf and pandas.Series.to_hdf5 functions have been
    used to save the data as hdf5, please use the complimentary functions
    pandas.read_hdf to retrieve the informations.

    A Logbook entry might also be associated, under Dataset.logbook_entry

    It is possible to add commentaries for each Dataset by using the
    Dataset.comment() and to specify some additional inf with the function
    Dataset.metadata()

    Each Dataset can be retrieved by using the function Dataset.unpickle()
    with argument the path of the saved Class.

    THE DATASETS CLASS IS MEANT TO BE READ VIA THE thorondor.gui CLASS !!
    """

    def __init__(self, df, path, name, savedir):
        """
        Initialiaze the Dataset class,
        some metadata can be associated as well
        """
        self.saving_directory = savedir
        self.df = df
        self.original_data = path
        self.name = name
        self.commentary = ""
        self.author = None
        self.instrument = None
        self.experiment = None
        self.purpose = None
        self.logbook_entry = None

        self.shifted_df = pd.DataFrame()
        self.reduced_df = pd.DataFrame()
        self.reduced_df_splines = pd.DataFrame()
        self.fit_df = pd.DataFrame()

        try:
            self.timestamp = datetime.utcfromtimestamp(
                os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
        except:
            print("timestamp of raw file not valid")
            self.timestamp = datetime.date.today()

        self.pickle()

    def pickle(self):
        """Use the pickle module to save the classes"""
        try:
            with open(
                f"{self.saving_directory}/"+self.name.split("~")[0]+".pickle",
                    'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        except PermissionError:
            print("""
                Permission denied, You cannot save this file because you \
                are not its creator. The changes are updated for this session \
                and you can still plot figures but once you exit the program, \
                all changes will be erased.
                """)

    @staticmethod
    def unpickle(file_path):
        """
        Use the pickle module to load the classes

        :param file_path: absolute path to pickle file
        """

        with open(f"{file_path}", 'rb') as f:
            return pickle.load(f)

    def metadata(self, **kwargs):
        """
        Add some metadata via this method

        :param kwargs: kwargs to store in file
        """

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.pickle()

    def comment(self, prompt, eraseall=False):
        """
        Precise if you want to erase the previous comments,
        type your comment as a string
        """
        try:
            if eraseall:
                self.commentary = ""
            self.commentary += prompt + "\n"
            print(self.commentary)
            self.pickle()

        except Exception as e:
            raise e

    def __repr__(self):
        try:
            if self.author == None:
                return f"{self.name}, created on the {self.timestamp}.\n"
            else:
                return "{}, created by {} on the {}, recorded on the instrument {}, experiment: {}.\n".format(
                    self.name, self.author, self.timestamp, self.instrument, self.experiment)
        except:
            return "Thorondor Dataset."

    def __str__(self):
        return repr(self)

    def to_hdf5(self, filename):
        """
        Save the data as hdf5 file, each group must be opened via the
        pandas.read_hdf() method.
        E.g.: pd.read_hdf("DatasetXXX.h5", "Dataframes/fit_df")
        The metadata is saved under the metadata attribute, see with
        pd.read_hdf("DatasetXXX.h5", "metadata")
        """
        try:
            StringDict = {keys: [
                value] for keys, value in self.__dict__.items() if isinstance(value, str)}
            DfDict = {keys: value for keys, value in self.__dict__.items(
            ) if isinstance(value, pd.core.frame.DataFrame)}
            SeriesDict = {keys: value for keys, value in self.__dict__.items(
            ) if isinstance(value, pd.core.series.Series)}

            StringDf = pd.DataFrame.from_dict(StringDict)

            for keys, values in DfDict.items():
                values.to_hdf(f"{filename}.h5", f"Dataframes/{keys}", mode="a")

            StringDf.to_hdf(f"{filename}.h5", "metadata", mode="a")

            # for keys, values in SeriesDict.items():
            #     values.to_hdf(f"{filename}.h5", f"Series/{keys}", mode = "w")

        except Exception as e:
            raise e

    def to_nxs(self):
        """
        hdf5 alias. Since thorondor only proceeds to the analysis of the
        data and does not directly handle the output of instruments, the NeXuS
        data format can be used to save the Dataset class, but the architecture
        follows the processed data file, metadata follows only if given by
        the user.

        Each dataframe used in thorondor is saved in
        root.NXentry.NXdata.<dataframe> as a table, the description of the table
        gives the column names.

        An error might raise due to fact that we use unicode character to write
        mu, this is not important.
        Uses pytables.
        """
        try:

            DfDict = {keys: value for keys, value in self.__dict__.items(
            ) if isinstance(value, pd.core.frame.DataFrame)}
            DfDictNotEmpty = {
                keys: values for keys, values in DfDict.items() if not DfDict[f"{keys}"].empty}

            with tb.open_file(f"{self.name}.nxs", "w", title=f"{self.name}, X-ray spectroscopy dataset, processed via thorondor.") as f:

                # Create nexus entry
                f.create_group(
                    "/", "NXentry", "Entry following nxs rules. Handle with pytables.")

                # Create NXdata
                f.create_group("/NXentry/", 'NXdata', "Contains all the data")

                # Create NXsample
                # f.create_group("/NXentry", "NXsample", "Inf. about sample")

                # Create NXinstrument
                # f.create_group("/NXentry", "NXinstrument", "Inf. about instrument")
                # f.create_group("/NXentry/NXinstrument", "NXdetector", "")
                # f.create_group("/NXentry/NXinstrument", "NXsource", "")

                # Create NXprocess
                f.create_group(
                    "/NXentry/", "NXprocess", """Processed via thorondor (see https://pypi.org/project/thorondor/)""")

                # Save the dataframes
                f.create_group("/NXentry/NXprocess",
                               "thorondor_dataframes", "")

                for DfName, DF in DfDictNotEmpty.items():
                    desc = np.dtype([(i, j) for (i, j) in (DF.dtypes.items())])
                    table = f.create_table(
                        "/NXentry/NXprocess/thorondor_dataframes", DfName, desc, DfName)
                    table.append(DF.values)

                # Create soft link towards data
                f.create_soft_link("/NXentry/NXdata", "data",
                                   "/NXentry/NXprocess/thorondor_dataframes/df")

                print(f)

        except Exception as e:
            raise e
