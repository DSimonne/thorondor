# Presentation of thorondor, a program for data analysis and treatment in NEXAFS

Authors : Simonne and Martini

#### The Program is meant to be imported as a python package, if you download it, please save the thorondor folder in ...\Anaconda3\Lib\site-packages
The installation command can be found here : https://pypi.org/project/thorondor/

To keep the package up to date, please use the following command : `pip install thorondor -U`

There are two main classes at the core of thorondor:

### The class `Dataset`
A new instance of the Dataset class will be initialized for each Dataset saved in the data folder. This object is then saved in the list "ClassList", attribute of the second class "GUI".
For each Dataset, all the different dataframes that will be created as well as specific information e.g. the edge jump E0 can be found as attributes of this class. Certain attributes are instanced directly with the class, such as:
* Name of Dataset
* Path of original dataset
* Timestamp

At the end of the data reduction, each Dataset should have at least three different data sets as attributes, saved as `pandas.DataFrame()`:
* Renamed data : Original data
* Shifted data : If one shifts the energy 
* Reduced data : If one applies some background reduction or normalization method 

A Logbook entry might also be associated, under `Dataset.logbook_entry`, this is done via the GUI, the logbook should be in the common excel formats.

It is possible to add commentaries for each Dataset by using the `Dataset.comment()` and to specify some additional information with the function `Dataset.metadata()`.

Each Dataset can be retrieved by using the function `Dataset.unpickle()` with the path of the saved Class as an argument. This is the recommended way to share the data, since it comes with all the metadata linked to the data collection and analysis.

### The class `Interface` (used to be `GUI`)
This  class is a Graphical User Interface (GUI) that is meant to be used to process nulerous XAS datasets, that focus on similar energy range and absorption edge.
There are two ways of initializing the procedure in a jupyter notebook:
* `GUI = thorondor.Interface()`; one will have to write the name of the data folder in which all his raw datasets are saved in a common data format (*.txt*, *.dat*, *.csv*, *.xlsx*).
* `GUI = thorondor.Interface.get_class_list(data_folder = "<yourdatafolder>")` ; if one has already worked on a folder and wishes to retrieve his work.

This class makes extensive use of the ipywidgets and is thus meant to be used with a jupyter notebook. Additional informations concerning the use of the Interface functions are provided in the "ReadMe" tab. One may also access method information through the python `help` built-in function.

All the different attributes of this class can also be exported in a single hdf5 file using the pandas `.to_hdf5` methods. They should then be accessed using the `read_hdf` methods.
A .nxs format support is in development.

The necessary Python packages are : numpy, pandas, matplotlib, ipywidgets, iPython, scipy, lmfit, emcee, thorondor and pytables.

### Citation and additional details

THORONDOR: software for fast treatment and analysis of low-energy XAS dat. Simonne, D.H., Martini, A., Signorile, M., Piovano, A., Braglia, L., Torelli, P., Borfecchia, E. & Ricchiardi, G. (2020). J. Synchrotron Rad. 27, https://doi.org/10.1107/S1600577520011388.

### FlowChart

![FlowChart](https://user-images.githubusercontent.com/51970962/81314649-aae0cd00-9089-11ea-9300-4c2e8ce47dc1.PNG)


### Please follow the following architecture when using the software


![Architecture1](https://user-images.githubusercontent.com/51970962/92746823-e36c1c80-f383-11ea-8850-79a7ab35b114.PNG)


### For users on Jupyter Lab, please follow this thread : https://stackoverflow.com/questions/49542417/how-to-get-ipywidgets-working-in-jupyter-lab

(perhaps outdated)