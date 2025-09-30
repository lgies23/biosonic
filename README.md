# BioSonic
A python package for bioacoustics


## Description

BioSonic is a python package for bioacoustics analysis. It's goal is to provide a solution for common workflows from normalization of files and basic acoustic feature extraction to extracting features commonly used in ML pipelines as well as pitch tracking with a user friendy function based structure and parametrization.


## Getting Started

### Dependencies

BioSonic is written to be lightweight and only relies on numpy (>= 1.26), scipy (>= 1.12) and pandas (>=2.3) for it's basic functionality. If you want plotting, this can be specified during pip installing:

```
pip install biosonic[plots]
```

For full functionality, the current dependencies are:

- matplotlib>=3.9.4
- numpy>=1.26.4
- pandas>=2.3.0
- scipy>=1.12
- praat-textgrids>=1.4.0

Python 3.10 and above are supported. Python 3.9 may be prone to some errors due to type checking but can be used with caution. 

### Installing

For now, clone the repository and run this command inside its root: 
```
pip install -e . 
```

### Executing

See the jupyter notebook `example_usage.ipynb` for demonstrations of different functionalities. 

## Authors

- Lena Gies (a12113965@unet.univie.ac.at)
- Tecumseh Fitch (tecumseh.fitch@unet.univie.ac.at)

## Acknowledgments and References

* Anikin A. 2019. Soundgen: an open-source tool for synthesizing nonverbal vocalizations. Behavior Research Methods, 51(2), 778-792.
* Boersma P. (1993) Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound. IFA Proceedings 17, 97–110.
* Childers DG, Skinner DP, Kemerait RC. (1977) The cepstrum: A guide to processing. Proc. IEEE 65, 1428–1443. https://doi.org/10.1109/PROC.1977.10747
* Klapuri A, Davy M. (2006) Signal processing methods for music transcription. New York: Springer. p.136
* Shannon C. E. (1948) A mathematical theory of communication. The Bell System Technical Journal XXVII.
* Sueur, J. (2018). Sound Analysis and Synthesis with R (Springer International Publishing). https://doi.org/10.1007/978-3-319-77647-7.
* Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272. https://doi.org/10.1038/s41592-019-0686-2.

  
* https://de.mathworks.com/help/signal/ref/spectralentropy.html accessed January 13th, 2025. 18:34 pm
* https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.entropy.html accessed May 20th 2025, 11:32 am
