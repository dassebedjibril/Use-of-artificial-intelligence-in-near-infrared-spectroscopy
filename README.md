## Use-of-artificial-intelligence-in-near-infrared-spectroscopy
This work has benefited from aid from the French State giving a favorable opinion from the government and which is part of a collaborative research initiative proposing to study the approaches relating to artificial intelligence, precisely that of the Deep Learning. It relies on a large dataset available online (https://esdac.jrc.ec.europa.eu/projects/lucas) and of which I would only study one part that will be necessary to achieve my goal which is to implement the Transfer Learning and testing its effectiveness in terms of performance, speed of convergence, and robustness.

### Near infrared spectroscopy (NIRS)

The frequencies that each medium absorbs
is specific to it, which allows the analysis of its composition and its properties

<div align="center">
    <img src="img.PNG" width="250px"</img> 
</div>

The study of spectra is called spectroscopy. To study the composition of a
middle, we illuminate it with a continuous spectrum and we note the different wavelengths
absorbed from which one can deduce the elements present

<div align="center">
    <img src="img1.PNG" width="250px"</img> 
</div>

The absorption of light at each of these wavelengths make up the “spectrum” of the sample. This spectrum can be made up of several hundred wavelengths for each of which we have measured the absorption of light.

### Signal processing Near infrared spectroscopy (NIRS)

The spectrum is characteristic of a sample because it gathers information (quantity
and characteristics) of each of its organic constituents (proteins, matter
fats, fibers, etc.). This wealth of information constitutes the advantage and the difficulty of
NIRS analysis: a lot of information is present in a spectrum, but it is
completely tangled! To overcome this difficulty, it is necessary to use
complex statistical methods, which will make it possible to link the spectra and the
chemical analyses: this is the `calibration` phase. Knowledge of the criteria
main characterization of the models makes it possible to quickly judge the quality of the
calibrations presented. Thereafter, we have the evaluation of the precision that we will have
during the practical use of the calibration which is the `validation`.


### Dataset

The NIRS has a large number of applications in the industrial field (chemistry,
pharmacy, agro-industries). For example, in the animal feed laboratory of the
CIRAD it is used to estimate the chemical composition of food samples,
fodder, products (meat), faeces (digestibility studies). Within the UMR
Agap Institute, it makes it possible to study the composition of sorghum grains, tubers
yams or rice leaves.

<div align="center">
    <img src="c.PNG" width="250px"</img> 
</div>

For our study, the FildSpec Near IR Spectrometer (ADS) was used with the lengths
wave between 350 and 2500 nm. The spectrum obtained can be visualized by looking at
the curve of the absorbance of the sample as a function of wavelength. We obtain
the following spectrum for our samples

<div align="center">
    <img src="d.PNG" width="250px"</img> 
</div>

For each dataset we used, we have the number of spectra as well
than the size in kilobytes:





LUCAS_SOC_cropland_Nocita is the learning game from which we will extract the weights
layers to transfer. It is used as a reference because it is the most common dataset.
complete (6111 spectra).
Each data set is divided into a calibration set (3/4 of the surrounding data) and a
validation dataset (1/4 of the data). The initial dataset split is
