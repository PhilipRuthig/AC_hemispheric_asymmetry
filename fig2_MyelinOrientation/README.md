Accompanying repository to Ruthig, Müller et al. (in prep)

This repository contains the code for the analysis pipeline of the orientation of myelinated structures in the mouse AC. It also contains code to reproduce Fig2 and Fig S5. In order to run the code, you will need to insert the data files from Bioimage archive (https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1077) to the folder data_from_bioimg_arx.

Then, the channels are split to retrieve the autofluorescence (C01) and the MBP channel (C04). Additionally, the channels are flipped vertically and horizontally for the left AC data and flipped horizontally only for the right AC. This was done as a pre-processing step using Fiji. 
