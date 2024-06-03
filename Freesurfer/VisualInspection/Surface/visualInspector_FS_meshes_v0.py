# Code to produce pictures from Freesurfer reconstructions for visual inspection
# It assumes a flattened folder organization 
# dataset
# ---- subj1
# -------- FS reconstruction folders (mri, surf, stats ...)
# ---- subj2
# -------- FS reconstruction folders (mri, surf, stats ...)
#
# runtime ~ 1 min 30s per subject 
#
# TO DO:
# 1) Other file organizations (BIDS, longitudinal)
# 2) improve speed. Update imshow() and tricontour data instead of rebuilding it
# 4) Input a csv with a list of selected subjects
# 5) Option to open a GUI (video or slide show) to open the pictures and annotate the evaluation in a csv
# 6) Other reconstructions
# 
# Victor B. B. Mello, April 2024 
# Support Center for Advanced Neuroimaging (SCAN)
# University Institute of Diagnostic and Interventional Neuroradiology
# University of Bern, Inselspital, Bern University Hospital, Bern, Switzerland.
#
# For the Newcastle+Rio+Bern (NewcarRiBe) collaboration
import argparse
import glob
import os
from multiprocessing.pool import Pool

import numpy as np
import matplotlib.pyplot as plt

import nibabel
import mne
from mne.transforms import apply_trans

from PIL import Image

def plot_func(input_list):
    # unpack input for parallel processing
    mri_slice = input_list[0]
    data_list = input_list[1] 
    
    # data for the plot
    data = data_list[0]    
    lhpial_rr_vox = data_list[1]
    lhpial_tris = data_list[2]

    rhpial_rr_vox = data_list[3]
    rhpial_tris = data_list[4]

    lhwhite_rr_vox = data_list[5]
    lhwhite_tris = data_list[6]
    
    rhwhite_rr_vox = data_list[7]
    rhwhite_tris = data_list[8]            
    
    subj = data_list[9]
    
    # save pictures for visual inspection
    plt.imshow(data[:,:,mri_slice].T,  cmap='gray', vmin=0, vmax=255)
    plt.tricontour(lhpial_rr_vox[:, 0], lhpial_rr_vox[:, 1], lhpial_tris, lhpial_rr_vox[:, 2], levels=[mri_slice], colors="r", linewidths=1.0, zorder=1,)
    plt.tricontour(rhpial_rr_vox[:, 0], rhpial_rr_vox[:, 1], rhpial_tris, rhpial_rr_vox[:, 2], levels=[mri_slice], colors="r", linewidths=1.0, zorder=1,)        
    plt.tricontour(lhwhite_rr_vox[:, 0], lhwhite_rr_vox[:, 1], lhwhite_tris, lhwhite_rr_vox[:, 2], levels=[mri_slice], colors="b", linewidths=1.0, zorder=1,)
    plt.tricontour(rhwhite_rr_vox[:, 0], rhwhite_rr_vox[:, 1], rhwhite_tris, rhwhite_rr_vox[:, 2], levels=[mri_slice], colors="b", linewidths=1.0, zorder=1,)                
    plt.savefig('surface_check_folder/'+subj+'/slice_{}.png'.format(mri_slice), dpi=320, format='png', transparent=False, bbox_inches='tight', pad_inches=0)
    plt.close()    

def make_images(path):
    subj = path.split("/")[-1]
    if not os.path.exists('surface_check_folder/'+subj):
        os.makedirs('surface_check_folder/'+subj)
    
    # T1 image
    t1 = nibabel.load(path+'/mri/orig.mgz')
    data = t1.get_fdata()
    affine = t1.header.get_vox2ras_tkr()

    # FS surfaces
    lhpial_rr_mm, lhpial_tris = mne.read_surface(path+'/surf/lh.pial')
    lhpial_rr_vox = apply_trans(np.linalg.inv(affine), lhpial_rr_mm)

    rhpial_rr_mm, rhpial_tris = mne.read_surface(path+'/surf/rh.pial')
    rhpial_rr_vox = apply_trans(np.linalg.inv(affine), rhpial_rr_mm)
    
    lhwhite_rr_mm, lhwhite_tris = mne.read_surface(path+'/surf/lh.white')
    lhwhite_rr_vox = apply_trans(np.linalg.inv(affine), lhwhite_rr_mm)
    
    rhwhite_rr_mm, rhwhite_tris = mne.read_surface(path+'/surf/rh.white')
    rhwhite_rr_vox = apply_trans(np.linalg.inv(affine), rhwhite_rr_mm)        

    # pack the data for parallel processing
    data_list = [data, lhpial_rr_vox, lhpial_tris, rhpial_rr_vox, rhpial_tris, lhwhite_rr_vox, lhwhite_tris, rhwhite_rr_vox,rhwhite_tris,subj]    
    input_list = [(i,data_list) for i in range(int(min(lhpial_rr_vox[:, 2])-1),int(max(rhpial_rr_vox[:, 2])+1),1)]
    
    # proccess multiple slices at the same time
    with Pool() as pool:
        res = pool.map(plot_func, input_list)

# Parser for running
parser = argparse.ArgumentParser(prog='visualInspector_FS_meshes', description='This code produces images for visual inspection of Freesurfer meshes')
parser.add_argument('-filepath', '--file', help='path to a specific subject folder for visual inspection', required = False)
parser.add_argument('-folderpath', '--folder', help='path to a folder containing the Freesurfer reconstructions from multiple subjects', required = False)
args = parser.parse_args()    

# create a folder for saving the pictures for visual inspection
if not os.path.exists('surface_check_folder'):
   os.makedirs('surface_check_folder')

if args.file == None and args.folder == None:
    print('Please give a path to a specific subject or multiple ones. For usage run: python3 visualInspector_FS_meshes_v2.py -h')
    raise SystemExit                                       

# deal with path to subj and path to a folder with multiple subjects
#
# Multiple subjects argument
if args.folder != None:
    path = args.folder
    subj_list = sorted(list(os.listdir(path)))
    path_list = [path +'/'+i for i in subj_list]

    for path_subj in path_list:
        make_images(path_subj)

# single subject argument
else:
    path = args.file
    make_images(path)

