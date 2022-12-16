# ADOPT-AP

Welcome to the model code of ADOPT-AP. No data is included in this repository, but we hope this code is useful in exploring how to couple water and human systems in a agent-based Python environment. Please note that this model is applied for agro-pastoralists in East Africa specifically - modifications are needed when applying it to other case studies, types of agents and their decisions.

## Overview

An overview of the proposed model framework can be found below. ADOPT-AP consists of three parts: 1) the environment, 2) socio-hydrological interactions and 3) human decision-making. The modelling framework integrates the DRYP hydrological model (https://github.com/AndresQuichimbo/DRYP) with the human behaviour component of the ADOPT model (Wens et al., 2020) and is built upon the ABM package Honeybees (https://github.com/jensdebruijn/honeybees). As the DRYP 1.0 is modified (in the ABM_connector.py especially), we have included the model in this repository as well. ADOPT-AP has been designed to simulate individual drought responses by agropastoralists (the agents) within their environment represented by the hydrological model output. Agropastoralists are here defined as households that grow crops, tend livestock or a combination of both as their main livelihood. The socio-hydrological interactions represent the feedbacks between agropastoralists and environmental land–water processes (i.e. water demand and grass and crop yield) of the drylands in Eastern Africa. Human decision-making is represented using PMT, a theory of decision-making under threat and implemented into the model by quantifying the factors that drive the intention to adapt to drought risk.

<img align="centre" src="https://github.com/istreefkerk/ADOPT-AP/blob/105b1ae1b26e4933874dcd3a1af8a228d670c9fd/Figure_1.jpg" width=60% height=60% >

## Research papers

The publication of ADOPT-AP can be found here:
https://doi.org/10.3389/frwa.2022.1037971

Publications of sub-models:

- DRYP 1.0 model (Quichimbo et al., 2021): https://doi.org/10.5194/gmd-14-6893-2021
- ADOPT model (Wens et al., 2020): https://doi.org/10.3389/frwa.2020.00015
