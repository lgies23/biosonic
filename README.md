# biosonic
A python package for bioacoustics

**This package is in active development. If you want to work with it, be aware that it is prone to bugs and the functionality might change. If you would like to collaborate, please reach out to us! We would love for this to become a comprehensive package for bioacoustics and a collaborative project.**


## Description

biosonic is a python package for bioacoustics analysis. It's goal is to provide a solution for common workflows from normalization of files and basic acoustic feature extraction to extracting features commonly used in ML pipelines as well as pitch tracking with a user friendy function based structure and parametrization.


## Getting Started

### Dependencies

BioSonic is written to be lightweight and only relies on numpy, scipy, and pandas for it's basic functionality. If you want plotting, this can be specified during pip installing:

```
pip install biosonic[plots]
```

For full functionality, the current dependencies are:

- numpy
- pandas
- scipy
- matplotlib
- praat-textgrids

Python 3.10 and above are supported.

### Installing

For full functionality, including plotting and praat-textgrid support run
```
pip install biosonic[all]
```

### Executing

Demonstrations of different functionalities can be found in the [documentation](https://biosonic.readthedocs.io/en/stable/example_usage.html).

## Authors

- Lena Gies (a12113965@unet.univie.ac.at)
- Tecumseh Fitch (tecumseh.fitch@unet.univie.ac.at)
- Yannick Jadoul (yannick.jadoul@uniroma1.it)

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
