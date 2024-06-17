# This function extracts the numeric data from FreeSurfer subjects into
# one table on a hemisphere and lobe basis.
#
# The code is a almost an exact translation of
# 
# https://github.com/cnnp-lab/CorticalFoldingAnalysisTools/tree/master
# 
# -> WM/GM contrast added
#
# Original Authors
# Yujiang Wang, September 2016 (extractMaster_Hemisphere.m)
# Tobias Ludwig & Yujiang Wang, September 2019
# Newcastle University, School of Computing, CNNP Lab (www.cnnp-lab.com)
#
# Victor B. B. Mello, November 2023 
# Support Center for Advanced Neuroimaging (SCAN)
# University Institute of Diagnostic and Interventional Neuroradiology
# University of Bern, Inselspital, Bern University Hospital, Bern, Switzerland.
#
# For the Newcastle+Rio+Bern (NewcarRiBe) collaboration

import numpy as np
import pandas as pd
import nibabel as nib
import pymeshlab
import trimesh
import argparse
import os
import sys
import csv
from multiprocessing.pool import Pool

import time

def partArea(vertices, faces, condition):
    # Area of a selected region
    # Dealing with the border: 1/3 triangle area if only one vertex is inside the ROI
    # 2/3 triangle area if 2 vertices are inside the ROI
    r_faces = np.isin(faces, condition)    
    fid = np.sum(r_faces,axis=1)
    area = 0
    array_area = trimesh.triangles.area(vertices[faces])
    array_area_corrected = np.zeros_like(array_area)
        
    for i in range(1,4,1):
        area = area + i*np.sum(array_area[fid==i])/3
        array_area_corrected[fid==i] = i*array_area[fid==i]/3

    return area, array_area_corrected

def compute_correction(initialv, initialf, scalar_label, label_selection, finalv, finalf, Kg):
    # Transfer selection from mesh to mesh by proximity
    ms_transfer = pymeshlab.MeshSet()
    morig = pymeshlab.Mesh(vertex_matrix=initialv,face_matrix=initialf,v_scalar_array=scalar_label)
    mfinal = pymeshlab.Mesh(vertex_matrix=finalv,face_matrix=finalf,v_scalar_array=Kg)

    ms_transfer.add_mesh(morig,'initial_mesh')
    ms_transfer.compute_scalar_transfer_vertex_to_face()    

    # loop to generate condsel for pymeshlab
    temp = ''
    for lbl in label_selection:
        temp = temp + '(fq == {}) || '.format(lbl)   
    str_qsel = '('+temp[:-3]+')'
    
    ms_transfer.compute_selection_by_condition_per_face(condselect = str_qsel)    
    ms_transfer.add_mesh(mfinal,'final_mesh')        
    ms_transfer.transfer_attributes_per_vertex(sourcemesh = 0, targetmesh = 1, selectiontransfer = True)

    ms_transfer.set_current_mesh(1)
    ms_transfer.generate_from_selected_vertices()
                
    return 4*np.pi/np.sum(ms_transfer.current_mesh().vertex_scalar_array())

# Improve
def transfer_label(initialv, initialf, scalar_label, finalv, finalf):
    # Transfer labels from mesh to mesh by proximity
    # label with the same label of the closest labeled vertex
    ms_transfer = pymeshlab.MeshSet()
    morig = pymeshlab.Mesh(vertex_matrix=initialv,face_matrix=initialf,v_scalar_array=scalar_label)
    mfinal = pymeshlab.Mesh(vertex_matrix=finalv,face_matrix=finalf)
    ms_transfer.add_mesh(morig,'initial_mesh')
    ms_transfer.add_mesh(mfinal,'final_mesh')        
    ms_transfer.transfer_attributes_per_vertex(sourcemesh = 0, targetmesh = 1, qualitytransfer = True)
    ms_transfer.compute_scalar_transfer_vertex_to_face()        
    ms_transfer.set_current_mesh(1)
    m = ms_transfer.current_mesh()
    
    return m.vertex_scalar_array()
    
def map_voxelvalue(pts_arr, volume):
    # get the voxel value from an array of indexes
    intarray = np.array(pts_arr, dtype=np.int32)
    label_pts = volume[intarray[:,0],intarray[:,1],intarray[:,2]].reshape(len(intarray),1)
    
    return label_pts

def do_convex_hull(vertices, faces):
    ms = pymeshlab.MeshSet()            
    m = pymeshlab.Mesh(vertex_matrix=vertices,face_matrix=faces)
    ms.add_mesh(m,'surface')
    ms.generate_convex_hull()
    ms.set_current_mesh(1)   
    ms.generate_resampled_uniform_mesh()    
    return ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()

def make_label_facebased(vertices, faces, label):
    # map vertex label into face label
    ms = pymeshlab.MeshSet()            
    m = pymeshlab.Mesh(vertex_matrix=vertices,face_matrix=faces, v_scalar_array=label)
    ms.add_mesh(m,'surface')
    ms.compute_scalar_transfer_vertex_to_face() 

    return ms.current_mesh().face_scalar_array() 
    
def extract_FSHemi_features(input_list):
    # unpack input
    filepath = input_list[0]
    args = input_list[1]    
    csv_output = args.outputpath
    
    hemisphere = ["r","l"]
    file_ID = filepath.split("/")[-1]

    result = [] 

    # import lobes_lut.csv to translate labels in lobes
    lut_lobes = pd.read_csv('lut_lobes.csv')
    lobes_list = ['Temporal','Parietal','Frontal','Occipital']
    lobes_labels = [list(lut_lobes['annotation'][ lut_lobes['lobe'] == lobe].values) for lobe in lobes_list]
    
    lobes_list.append('Hemisphere')
    all_lbl = list(lut_lobes['annotation'])
    all_lbl.remove(-1)    
    lobes_labels.append(all_lbl)
        
    labels_dict = dict(zip(lobes_list,lobes_labels)) 
    for hemi in hemisphere:
        error = False
        # Reading FreeSurfer's reconstruction
        try:
            # Thickness
            thickness = nib.freesurfer.io.read_morph_data(filepath+"/surf/"+hemi+"h.thickness")
            # annotation
            annot, ctab, labels_name = nib.freesurfer.io.read_annot(filepath+"/label/"+hemi+"h.aparc.annot")            
            # Pial surface
            pialv, pialf = nib.freesurfer.io.read_geometry(filepath+"/surf/"+hemi+"h.pial")
            # Smooth outer pial surface
            smoothpialv, smoothpialf = nib.freesurfer.io.read_geometry(filepath+"/surf/"+hemi+"h.pial-outer-smoothed")
            # WM surface
            whitev, whitef = nib.freesurfer.io.read_geometry(filepath+"/surf/"+hemi+"h.white")

            # MRI image
            mri_img = nib.load(filepath+"/mri/norm.mgz")
            intensity_val = mri_img.get_fdata()
            ras_tkr2vox = np.linalg.inv(mri_img.header.get_vox2ras_tkr())
            voxel_size = mri_img.header.get_zooms()[0]

            # convex hull of the pial surface
            pialv_ch, pialf_ch = do_convex_hull(pialv,pialf)
            ch_label = transfer_label(pialv, pialf, annot, pialv_ch, pialf_ch)
            smooth_label = transfer_label(pialv, pialf, annot, smoothpialv, smoothpialf)

            # thickness labeling facebased
            t_facebased = make_label_facebased(pialv,pialf,thickness)

            # calculate vertex defect in convex hull 
            # vertex deffects -> Kg = 2pi - sum angles
            ch_Kg = trimesh.curvature.vertex_defects(trimesh.Trimesh(pialv_ch,pialf_ch))

            # make wm/gm contrast map
            # from FS surface coordinate to voxel space
            whitev_voxel_space = nib.affines.apply_affine(ras_tkr2vox, whitev)
            # create mesh in the voxel space
            mwhite = pymeshlab.Mesh(vertex_matrix=whitev_voxel_space,face_matrix=whitef)
            ms = pymeshlab.MeshSet()
            ms.add_mesh(mwhite,'surf')
            ms.compute_normal_per_face()
            ms.apply_normal_normalization_per_face()
            ms.meshing_invert_face_orientation()
        
            # Sample voxels
            # WM: Intensity 1 mm inside WM (face normal direction)
            # GM: Intensity 35% of the thickness inside GM (face normal direction)
            # normal pointing outside WM
            white_fnormals = ms.current_mesh().face_normal_matrix()
            center = np.mean(whitev_voxel_space[whitef],axis=1)
            wm = map_voxelvalue(center-white_fnormals,intensity_val)
            gm = map_voxelvalue(center+white_fnormals*t_facebased.reshape(-1,1)*0.35/voxel_size,intensity_val)
            # deal with the 0 intensity values
            # WM/GM contrast definition: https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.22103 
            eps = 1e-10
            temp_contrast = wm/(gm+eps)
            temp_contrast_normalized = (wm-gm)/(gm+wm)
            
            for idx, lobe in enumerate(lobes_list):
                data_dict = {}
                # vertex based selection
                label_selection = np.where(np.isin(annot, labels_dict[lobe]))
                
                # morphometrics
                pial_area, pial_area_array = partArea(pialv, pialf, condition = label_selection)
                smooth_label_selection = np.where(np.isin(smooth_label, labels_dict[lobe]))
                smoothpial_area, smoothpial_area_array = partArea(smoothpialv, smoothpialf, condition = smooth_label_selection)
                
                ch_label_selection = np.where(np.isin(ch_label, labels_dict[lobe]))
                pial_area_ch, pial_area_array_ch = partArea(pialv_ch, pialf_ch, condition = ch_label_selection)                                
                
                white_area, white_area_array = partArea(whitev, whitef, condition = label_selection)

                # Region mean thickness        
                w = (pial_area_array/np.sum(pial_area_array) + white_area_array/np.sum(white_area_array))/2
                avg_thickness = np.sum(w*t_facebased)
                
                # compute correction for the areas
                correction_Ig = compute_correction(pialv, pialf, annot, labels_dict[lobe], pialv_ch, pialf_ch, ch_Kg)

                # compute the contrast
                lobe_contrast = temp_contrast[label_selection]
                mean_contrast = np.mean( lobe_contrast[ lobe_contrast < 100 ] )
                lobe_contrast_normalized = temp_contrast_normalized[label_selection]
                mean_contrast_normalized = np.mean( lobe_contrast_normalized )

                # select voxels hit by the surface
                varray_voxel = np.array(nib.affines.apply_affine(ras_tkr2vox, whitev)[label_selection],dtype=np.int32) 
                sorted_v =  varray_voxel[np.lexsort(varray_voxel.T),:]
                row_mask = np.append([True],np.any(np.diff(sorted_v,axis=0),1))

                voxels = len(sorted_v[row_mask])
                
                # output data                
                data_dict['subj'] = file_ID
                data_dict['hemi'] = hemi
                data_dict['region'] = lobe
                data_dict['pial_area'] = pial_area
                data_dict['convexhull_pial_area'] = pial_area_ch
                data_dict['smooth_pial_area'] = smoothpial_area
                data_dict['wm_area'] = white_area
                data_dict['thickness'] = avg_thickness
                data_dict['mean_contrast'] = mean_contrast            
                data_dict['mean_contrast_normalized'] = mean_contrast_normalized                            
                data_dict['Ig_correction'] = correction_Ig
                data_dict['voxels'] = voxels                                            
                result.append(data_dict)
                        
        except FileNotFoundError:
            error = True
            print("WARNING: Missing reconstruction file from subject {}".format(file_ID), flush = True)
            break
            
        except KeyboardInterrupt:
            print("The programm was terminated manually!")
            raise SystemExit                                       

    if error == False:
        file_exists = os.path.isfile(args.outputpath)
        with open (csv_output, 'a') as csvfile:
            headers = list(data_dict.keys())
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)

            if not file_exists:
                writer.writeheader()
            writer.writerows(result)                        
                        
if __name__ == '__main__':
    # arguments from the shell: path to FS rec and the outputfile desired
    parser = argparse.ArgumentParser(prog='extract_FreeSurfer_Hemi_features', description='This code extracts from a Freesurfer reconstruction cortical morphological variables and WM/GM contrast values to a csv file in a hemisphere base ')
    parser.add_argument('-filepath', '--path', help='path to the folder containing the Freesurfer reconstructions from multiple subjects', required = True)
    parser.add_argument('-outputfile', '--outputpath', default = 'morphometric_data.csv', help='Path to the csv output file. Default is morphometric_data.csv to be saved at the running directory', required = False)
    args = parser.parse_args()

    # list of subjects packed together with args inside a tuple for the multiprocessing Map
    fs_files_path = args.path
    subj_list = os.listdir(fs_files_path)
    input_list = [(fs_files_path +'/'+i, args) for i in subj_list]

    # create a log file for the corrupted IDs
    log_file = open("logfile.log","w")
    sys.stdout = log_file    
    
    # proccess multiple subjects at the same time
    with Pool() as pool:
        res = pool.map(extract_FSHemi_features, input_list)
 
    log_file.close()        
