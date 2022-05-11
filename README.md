# Hello world

Welcome to the README about Thorondor, this version is specialized with XPS data, and was created to analyze the data collected at the B07 beamline, in Diamond.

Below, I will detail how to use the GUI, and what kind of workflow can be followed.

If this tool helped you, feel free to cite the paper about [`thorondor`](https://doi.org/10.1107/S1600577520011388).

# Loading data

Currently, only the B07 beamline is supported. For future support of other beamlines, feel free to look at the `DiamondDataset` class and to copy its architecture.

At B07, an experiment consists in collecting many surveys, for a given incoming photon energy. A survey has a specific:
* Name, e.g. "N 1s",
* Acquisition range, we will come back to that later,
* Number of iterations, more iterations take longer but end up reducing the Signal/Noise ratio of the data, providing that the experiment is stable,
* Counting time,
* Number of points.

Let's take the example of a N 1s survey, with 10 iterations, a counting time of 1 second, and an incoming photon energy of 700 eV.

The incoming photons excite the electronic states of the sample so that electrons are emitted, thanks to different process, with different kinetic energy.
The kinetic energy of the electron can vary from 0 (the electrons are not emitted) to a maximum energy that cannot excess the incoming photon energy (the photons then transfered all their energy to the electrons).

The kinetic energy of the electron, $E_k$, corresponds to:

$E_k = E_p - W - E_b$

with:
* $E_p$, the incoming photon energy
* $W$, the work function of the detector (energy needed to 'see' the electron), usually a few eVs.
* $E_b$, the binding energy of the electron, that corresponds to its electronic state in the sample, these are [tabulated values](xdb.lbl.gov/Section1/Table_1-1.pdf).

Thanks to the electron analyzer (detector), we are able to 'count' the number of electrons that are emitted for each kinetic energy. Translating this to the binding energy of those electrons:

$E_b = E_p - W - E_k$

We can quantify the electrons emitted for each binding energy, for example around the binding energy of the N 1s electronic state ($409.9$ eV), we should count a lot of electrons if the sample contains Nitrogen ! Otherwise the amount of electrons counted should be equal to zero.

Therefore, we fix the detection range around the binding energy of N 1s, let's take $[400, 420]$ eV, and do 201 steps (each step is then $0.1$ eV).

**This N 1s dataset can be resumed as follow:**

| Incoming photon energy | Electron binding energy detection range | Counting time | Number of points | Number of iterations |
|------------------------|---------------------------------|---------------|------------------|---|
| $700$ eV | $[400, 420]$ eV | $1.0$ s | 201 | 10 |

The resulting dataset will have several other attributes:
* `spectrum`: the intensity summed over all the iterations, its shape corresponds to the number of points, here $(1, 201)$.
* `spectra`: an array that corresponds to the electron count for each binding energy, but with a third dimension, the iteration number. Its shape corresponds to the number of points times the iteration number, here $(1, 10, 201)$.
* `binding_energy`: its shape corresponds to the number of points, here $(1, 201)$.
* `kinetic_energy`: its shape corresponds to the number of points, here $(1, 201)$.

Each `DiamondDataset` has multiple surveys. These are detailed when loading the data in `thorondor`. For example, i nthe image below we can see that the `b07-1-61845`  dataset has the surveys $Ef_250$ and $Pt4f_Al2p_250$.

![index](https://user-images.githubusercontent.com/51970962/167739162-d2bc7cd9-1fe1-438b-b79f-ceb7672e3182.png)

If we want detailsa bout each dataset, we can use the `View data` tab. For each dataset, the surveys are detailed.

![view_data](https://user-images.githubusercontent.com/51970962/167739175-450774d2-edfb-4b6b-ab14-65fe969a5b32.png)

## Peak broadening

Due to the electron analyzer, to the monochromator and to the finite lifetime of the core-hole excitation, the electrons are not only emitted for a specific energy but rather distributed over a range around a binding energy. For metals, this distribution follows the [Doniach Sunjic distribution](http://www.casaxps.com/help_manual/line_shapes.htm).

The image below details the peak broadening due to the electron analyzer.

![broad](https://user-images.githubusercontent.com/51970962/167739133-afedae99-8c4f-4dee-99e8-846ad894d60a.png)

## Example of XPS peak
The image below shows two peaks that correspond to the Pt 4f signal. It can be obtained via the `DiamondDataset.plot_data()` function

![peak example](https://user-images.githubusercontent.com/51970962/167739165-5c9624af-0459-4c2b-9fb6-ad63e1647744.png)

## Example of transition visible via a high number of iterations

On the image below, we can see a peak appear when introducing Oxygen in the sample chamber, which shows that the gas has an effect on the surface of the sample.

This image can be obtained via the `DiamondDataset.plot_iterations()` function

![transient](https://user-images.githubusercontent.com/51970962/167739167-a1266a19-4cc4-4f84-b8b0-f2b19c562774.png)

# Data reduction

## Peak shifts

If the surface of the sample is electrically insulating, then the emission of electrons causes a positive charge to accumulate at the surface. There is then a shift in the electron energy due to the sample's electromagnetic field.  It causes the peaks in the spectrum to shift to high binding energies and become distorted.

A rigid shift can be corrected by recording the Fermi level for metals.

The Fermi level of a solid-state body is the thermodynamic work required to add one electron to the body. The Fermi level does not include the work required to remove the electron from wherever it came from.

In band structure theory, used in solid state physics to analyze the energy levels in a solid, the Fermi level can be considered to be a hypothetical energy level of an electron, such that at thermodynamic equilibrium this energy level would have a 50% probability of being occupied at any given time.

In other words, the Fermi level is the surface of the sea at absolute zero where no electrons will have enough energy to rise above the surface.

Therefore, we can set the binding energy of the Fermi level at zero. It corresponds to the region in the spectra where we start to detect electrons.

![fermi](https://user-images.githubusercontent.com/51970962/167739144-4f98da24-642f-4329-be0a-d6d737af9639.png)

## Peak normalization

To compare peaks between each other, you may normalize the intensity over a data range. This will set the average value over this range to 1.

![norm](https://user-images.githubusercontent.com/51970962/167739164-765c844f-6b62-4a2d-a3f1-de4277795c3f.png)

# Peak fitting

In the image below, you can see how one may fit the experimental data with `thorondor` in the fitting tab.

![fitting1](https://user-images.githubusercontent.com/51970962/167739149-43f0b8eb-7365-4db0-b97c-1268ffcef185.png)
![fitting2](https://user-images.githubusercontent.com/51970962/167739152-6128d42c-bfed-4a92-80f6-510acd7b0b3f.png)

With the result of the fit below, each component is displayed as well as the residuals.

![fitting3](https://user-images.githubusercontent.com/51970962/167739155-ac4f0ee9-07f0-4792-8010-5b8313f2ae78.png)

The fitting results are then saved as `self.result`

![fitting4](https://user-images.githubusercontent.com/51970962/167739159-6f188d26-8426-419d-bd9b-3f78317b3014.png)

It is possible to link parameters via an expression, see [documentation](https://lmfit.github.io/lmfit-py/constraints.html)

```python
GUI.pars["P1_sigma"].expr = "P0_sigma"
GUI.pars["P1_gamma"].expr = "P0_gamma"
```

You can also load the results from another fit in the GUI:

```python
GUI.pars = GUI.class_list[0].dataframexxxx_fit_result.params.copy()
```
### Fitting statistics
#### How good is our fit ?
Quantified by the `chisqr` attribute, and other fitting criteria. The meaning of each criterion is detailed [here](https://lmfit.github.io/lmfit-py/fitting.html#goodness-of-fit-statistics).

![image](https://user-images.githubusercontent.com/51970962/167848858-784da4d2-f2a0-40ab-88c1-85a00cfe8bba.png)

Another common criteria is the R-factor that simply compares the ratio between the square of the sum of the residuals and the square of the sum of the data. If the fit is perfect, the sum of the residual (and thus this ratio) should tend towards 0.

$R = 1 - 100 * \frac{\sum_{i=0}^N (y_i - yf_i)^2}{\sum_{i=0}^N y_i^2}$

where $y_i$ is the observed data, and $yf_i$ is the value of the fit for $i$ \in $N$ bins $(x_i, y_i)$.

The chi square test can also be computed with the [`scipy.stats.chisquare()`](https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/scipy.stats.chisquare.html) method.

#### How accurate are our parameters final value ? How much does each parameter correlate with the others ?

After a fit using the `least_squares()` method has completed successfully, standard errors for the fitted variables and correlations between pairs of fitted variables are automatically calculated from the covariance matrix.
For other methods, the calc_covar parameter (default is `True`) in the Minimizer class determines whether or not to use the  `numdifftools` package to estimate the covariance matrix.

* The standard error (estimated error-bar) goes into the `stderr` attribute of the Parameter.
* The correlations with all other variables will be put into the `correl` attribute of the Parameter – a dictionary with keys for all other Parameters and values of the corresponding correlation.

In some cases, it may not be possible to estimate the errors and correlations. For example, if a variable actually has no practical effect on the fit, it will likely cause the covariance matrix to be singular, making standard errors impossible to estimate. Placing bounds on varied Parameters makes it more likely that errors cannot be estimated, as being near the maximum or minimum value makes the covariance matrix singular. In these cases, the errorbars attribute of the fit result (Minimizer object) will be False.



# Data visualisation
![vis](https://user-images.githubusercontent.com/51970962/167739176-8823768e-ad35-43af-9b30-5d83f9ba01ea.png)
