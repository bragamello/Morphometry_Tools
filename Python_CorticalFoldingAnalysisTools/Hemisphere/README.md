This function extracts the numeric data from multiple FreeSurfer subjects into one table on a hemisphere-basis. It assumes a flattened file organization

Example:
 derivatives
     ---- pipeline
         ---- subj1_ses1
         ---- subj1_ses2
         ---- subj2_ses1

Usage: python3 extract_FreeSurferHemi_features.py -filepath path2data -outputfile mydata.csv

Arguments
-filepath PATH, --path PATH
                        path to the folder containing the Freesurfer
                        reconstructions from multiple subjects
-outputfile OUTPUTPATH, --outputpath OUTPUTPATH
                        Path to the csv output file. Default is
                        morphometric_data.csv to be saved at the running
                        directory

Original Authors
Yujiang Wang, September 2016 (extractMaster_Hemisphere.m)
Tobias Ludwig & Yujiang Wang, September 2019
Newcastle University, School of Computing, CNNP Lab (www.cnnp-lab.com)

Victor B. B. Mello, November 2023 
Support Center for Advanced Neuroimaging (SCAN)
University Institute of Diagnostic and Interventional Neuroradiology
University of Bern, Inselspital, Bern University Hospital, Bern, Switzerland.
For the Newcastle+Rio+Bern (NewcarRiBe) collaboration
