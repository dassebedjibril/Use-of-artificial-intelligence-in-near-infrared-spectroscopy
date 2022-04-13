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
    <img src="c.PNG" width="400px"</img> 
</div>

For our study, the FildSpec Near IR Spectrometer (ADS) was used with the lengths
wave between 350 and 2500 nm. The spectrum obtained can be visualized by looking at
the curve of the absorbance of the sample as a function of wavelength. We obtain
the following spectrum for our samples

<div align="center">
    <img src="d.png" width="400px"</img> 
</div>

For each dataset we used, we have the number of spectra as well
than the size in kilobytes:


<div align="center">
    <img src="img2.PNG" width="450px"</img> 
</div>


LUCAS_SOC_cropland_Nocita is the learning game from which we will extract the weights
layers to transfer. It is used as a reference because it is the most common dataset.
complete (6111 spectra).
Each data set is divided into a calibration set (3/4 of the surrounding data) and a
validation dataset (1/4 of the data). The initial dataset split is performed using the Kennard-Stone algorithm. For LUCAS_SOC_Cropland_6111_Nocita for example, here are the generated files:

<div align="center">
    <img src="img3.PNG" width="450px"</img> 
</div>


### Model description

The model we used for this project is the convolutional neural network.
First, a convolutional neural network or convolutional neural network
(CNN) considered model-based machine learning tools
training data. They were developed by LeCun et al. in 1998 as a class
acyclic artificial neural networks (feed-forward). They are now one of
most important deep learning architectures, and they have been applied for
many tasks in different fields of research. It is a neural network
in which the connecting pattern between neurons is inspired by the visual cortex of
animals. Their operation is inspired by biological processes, they consist of a
multilayer stacking of perceptrons, the purpose of which is to preprocess small quantities
information. Experimental results conducted on spectroscopic data sets
show the interesting capabilities of 1D-CNN methods (CNN architectures at one
dimension) proposed. This network will take into account the application of 1D filters on the layers
convolution with the use of 1D input data.

CNNs architectures are the most used in learning approaches in depth. They are generally composed of an input layer, several hidden layers (convolution, pooling, and fully connected layers) and an output layer. 1D-CNN has an input layer and 1D filters on input layers. Convolution adapted to the one-dimensional spectrum of our data. In this study, we used the 1D-CNN architecture because the structure is simple and well described.

<div align="center">
    <img src="i2.JPG" width="450px"</img> 
</div>


### Results and discussion

This Project has three facets: one on the implementation of a Deep Learning code,
one on a performance analysis and finally one on the performance comparison
without and with Transfer Learning. This section is divided into three main parts which
describe.

#### General method

When training a network on a data set, this set must be divided into 2 parts: a
first for the learning phase (training), a second for the validation phase.
Indeed, it is possible for a network to over-learn the learning set. If so we do
predictions on this same game, the results will seem very good but when a
new dataset the results may have a clear difference. This phenomenon
is called overfitting, and to be able to detect it, we need the validation set
(so distinct from the learning game). Fortunately several effective techniques
exist to avoid overfitting, including: cross validation (cross validation in
French), Add training data, Remove features, Methods of
adjustments.
Unlike classic validation, where the data is divided in two, in cross
validation we divide the training data into several groups. The idea is then
to train the model on all the groups except one and to validate the training on this
latest. If we have 𝑘 groups, we will train the model 𝑘 times with each time a new
testing group. This cross-validation technique is called 𝑘 − 𝑓𝑜𝑙𝑑. We have
taken 𝑘 = 3. This technique helps to select the right machine learning models.

The use of all these techniques has allowed us to improve performance and with
the biggest dataset 𝐿𝑈𝐶𝐴𝑆_𝑆𝑂𝐶_𝑐𝑟𝑜𝑝𝑙𝑎𝑛𝑑 (of size 628 𝑀𝑜) we got
The following scores: 𝑅𝑀𝑆𝐸_𝐶𝑎𝑙𝑖𝑏𝑟𝑎𝑡𝑖𝑜𝑛 = 4.74, 𝑅𝑀𝑆𝐸_𝑉𝑎𝑙𝑖𝑑𝑎𝑡𝑖𝑜𝑛 = 3.88, 𝑅2_𝐶𝑎𝑙𝑖𝑏𝑟𝑎𝑡𝑖𝑜𝑛 = 0.91, 𝑅2_𝑉𝑎𝑙𝑖𝑑𝑎𝑡𝑖𝑜𝑛 = 0.70.

#### Transfer learning
