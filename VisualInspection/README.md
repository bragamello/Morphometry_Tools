Code to produce pictures from Freesurfer reconstructions for visual inspection. It assumes a flattened folder organization 

Example dataset
---- subj1

-------- FS reconstruction folders (mri, surf, stats ...)

---- subj2

-------- FS reconstruction folders (mri, surf, stats ...)

runtime ~ 1 min 30s per subject 

TO DO:
1) Other file organizations (BIDS, longitudinal)

2) improve speed. Update imshow() and tricontour data instead of rebuilding it

4) Input a csv with a list of selected subjects

5) Option to open a GUI (video or slide show) to open the pictures and annotate the evaluation in a csv

6) Other reconstructions
 
Victor B. B. Mello, April 2024 

Support Center for Advanced Neuroimaging (SCAN), University Institute of Diagnostic and Interventional Neuroradiology, 
University of Bern, Inselspital, Bern University Hospital, Bern, Switzerland.
# For the Newcastle+Rio+Bern (NewcarRiBe) collaboratio
