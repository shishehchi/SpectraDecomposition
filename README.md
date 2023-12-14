# SpectraDecomposition
Dissection of composite spectra of galaxies into individual components using deep supervised learning techniques.


# Summary

Recent advancements in galaxy research, particularly through Integral Field Unit (IFU) surveys like MaNGA, have significantly deepened our understanding of the diverse stellar populations and physical processes within galaxies. These surveys provide more precise measurements of key parameters such as star formation rates, thereby reducing uncertainties in our study of galactic complexities. However, the limited scope of IFU surveys, focusing primarily on local universe galaxies, restricts their broader application. In contrast, single-slit spectroscopic surveys like SDSS, which offer an averaged view by capturing integrated light from galaxies, face difficulties in accurately extracting specific details due to the overlapping contributions from different regions. To bridge such gaps, we have considered an innovative two-component scenario in which a model decomposes SDSS spectra into Active Galactic Nuclei (AGN) and Star-Forming (SF) components, utilizing MaNGA spectra for training. This approach, which leverages deep supervised learning techniques trained on IFU survey data, enables the dissection of composite spectra into individual components, providing a clearer quantification of each population’s contribution. This two-component decomposition system has the potential to extend to multicomponent and more complicated scenarios. Our results show that removing star-forming ’contamination’ from AGN galaxies can position the galaxies to a more limited area on BPT diagrams.

# The deep decomposition model
Our Deep Decomposition Model (DDM) consists of a model with five regressors, each with its own loss function (i.e., five outputs). A detailed information of the
structure of the model is below:

<img src="images/model_architecture.png" width="100%">
