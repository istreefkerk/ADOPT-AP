# ADOPT-AP

Welcome to the model code of ADOPT-AP - we hope this code is useful in exploring how to couple water and human systems in a agent-based Python environment. Please note that this model is applied for agro-pastoralists in East Africa specifically - modifications are needed when applying it to other case studies, types of agents and their decisions. Please see the sources of the input data in the publications outlined below.

## Overview

An overview of the proposed model framework can be found below. ADOPT-AP consists of three parts: 1) the environment, 2) socio-hydrological interactions and 3) human decision-making. The modelling framework integrates the DRYP hydrological model (https://doi.org/10.5281/zenodo) with the human behaviour component of the ADOPT model (Wens et al., 2020) and is built upon the ABM package Honeybees (https://github.com/jensdebruijn/honeybees). As DRYP 2.0 is modified (in the ABM_connector.py especially), we have included the model in this repository as well. ADOPT-AP has been designed to simulate individual drought responses by agropastoralists (the agents) within their environment represented by the hydrological model output. Agropastoralists are here defined as households that grow crops, tend livestock or a combination of both as their main livelihood. The socio-hydrological interactions represent the feedbacks between agropastoralists and environmental land–water processes (i.e. water demand and grass and crop yield) of the drylands in Eastern Africa. Human decision-making is represented using PMT, a theory of decision-making under threat and implemented into the model by quantifying the factors that drive the intention to adapt to drought risk.

<p align="center">
  <img src="https://github.com/istreefkerk/ADOPT-AP/blob/105b1ae1b26e4933874dcd3a1af8a228d670c9fd/Figure_1.jpg" width=60% height=60% >
</p>

## Research papers

The publications of ADOPT-AP can be found here:
- https://doi.org/10.3389/frwa.2022.1037971
- https://doi.org/10.1016/j.ijdrr.2025.105309
- https://doi.org/10.5194/egusphere-2024-2382

Publications of sub-models:

- DRYP 2.0 model (Quichimbo et al., 2025): https://doi.org/10.5194/egusphere-2025-5316
- ADOPT model (Wens et al., 2020): https://doi.org/10.3389/frwa.2020.00015
