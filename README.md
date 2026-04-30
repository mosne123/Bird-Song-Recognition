# Bird-Song-Recognition
Bird Song Recognition project based on TinyML on embedded Photon2 platform

This repository contains files needed for recording over PDM mic data of bird calls of varying species and also contains the
code for running a local instance of machine learning model to distinguish between the different bird calls.

The repo is split between various folder for easy navigation to the underlying files needed to run the machine learning project.

The 'data' folder contains both raw data of bird calls, but also that data chopped into smaller pieces and recorded via PDM mic by Photon2
of 30 seconds playtime each. This ensures that the training of a model based on determining the different bird calls also factor in caracteristics 
of the PDM mic itself.

The PDM-data-sample folder contains a cpp source file under 'src' folder which functions as the main code for sampling the data for training purposes.
Under the same folder, the tcp server file is found inside 'Microphone_PDM-main/server' folder and is needed for communication between Photon2 and PC to 
store bird call data.

To run the server, simply type:
	``` node server.js --rate 16000 --bits 16 ```
