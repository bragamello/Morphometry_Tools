# This function extracts the numeric data from multiple FreeSurfer subjects into
# one table on a hemisphere-basis.
#
# It assumes a flattened file organization
#
# derivatives
#     ---- pipeline
#         ---- subj1_ses1
#         ---- subj1_ses2
#         ---- subj2_ses1
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
import nibabel as nib
import pymeshlab
import trimesh
import argparse
import os
import sys
import csv
from multiprocessing.pool import Pool

def partArea(vertices, faces, condition):
    # Area of a selected region
    # Dealing with the border: 1/3 triangle area if only one vertex is inside the ROI
    # 2/3 triangle area if 2 vertices are inside the ROI
    r_faces = np.isin(faces, condition)    
    fid = np.sum(r_faces,axis=1)
    area = 0
    array_area = trimesh.triangles.area(vertices[faces])
    
    for i in range(1,4,1):
        area = area + i*np.sum(array_area[fid==i])/3

    return area

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

    return ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()

def make_label_facebased(vertices, faces, label):
    # map vertex label into face label
    ms = pymeshlab.MeshSet()            
    m = pymeshlab.Mesh(vertex_matrix=vertices,face_matrix=faces, v_scalar_array=label)
    ms.add_mesh(m,'surface')
    ms.compute_scalar_transfer_vertex_to_face() 

    return ms.current_mesh().face_scalar_array() 
    
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

def extract_FSHemi_features(input_list):
    # unpack input
    filepath = input_list[0]
    args = input_list[1]    
    csv_output = args.outputpath
    
    hemisphere = ["r","l"]
    file_ID = filepath.split("/")[-1]

    result = [] 
        
    for hemi in hemisphere:
        data_dict = {}
        error = False
        # Reading FreeSurfer's reconstruction
        try:
            # Thickness
            thickness = nib.freesurfer.io.read_morph_data(filepath+"/surf/"+hemi+"h.thickness")
                        
            # Pial surface
            pialv, pialf = nib.freesurfer.io.read_geometry(filepath+"/surf/"+hemi+"h.pial")
                                              
            # WM surface
            whitev, whitef = nib.freesurfer.io.read_geometry(filepath+"/surf/"+hemi+"h.white")
            
            # MRI image
            mri_img = nib.load(filepath+"/mri/norm.mgz")
            intensity_val = mri_img.get_fdata() 
            ras_tkr2vox = np.linalg.inv(mri_img.header.get_vox2ras_tkr())
            voxel_size = mri_img.header.get_zooms()[0]

            # Pure morphometrics from the FreeSurfer Meshes
            # units: mm
            # pial surface
            array_pialarea = trimesh.triangles.area(pialv[pialf])        
            pial_total_area = np.sum(array_pialarea)
            pial_volume = trimesh.triangles.mass_properties(pialv[pialf])['volume']

            # convex hull of the pial surface
            pialv_ch, pialf_ch = do_convex_hull(pialv,pialf)
            pial_convexhull_area = np.sum(trimesh.triangles.area(pialv_ch[pialf_ch]))                
            pial_convexhull_volume = trimesh.triangles.mass_properties(pialv_ch[pialf_ch])['volume']

            # WM surface
            array_whitearea = trimesh.triangles.area(whitev[whitef])        
            white_total_area = np.sum(trimesh.triangles.area(whitev[whitef]))
            white_volume = trimesh.triangles.mass_properties(whitev[whitef])['volume']        

            # Find the cc, brain stem ... to subtract the areas
            cc_pial_area = partArea(pialv, pialf, condition = np.where(thickness==0))
            cc_white_area = partArea(whitev, whitef, condition = np.where(thickness==0))

            # mean thickness
            t_facebased = make_label_facebased(pialv,pialf,thickness)        
            w = (array_pialarea[t_facebased != 0]/np.sum(array_pialarea[t_facebased != 0]) +  array_whitearea[t_facebased != 0]/np.sum(array_whitearea[t_facebased != 0]))/2
            avg_thickness = np.sum(w*t_facebased[t_facebased != 0])

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
            eps = 1e-10
            temp_contrast = wm/(gm+eps)
            mean_contrast = np.mean(temp_contrast[temp_contrast<100])
            
            temp_contrast_normalized = (wm-gm)/(gm+wm)
            mean_contrast_normalized = np.mean(temp_contrast_normalized)
        
        except FileNotFoundError:
            error = True
            print("WARNING: Missing reconstruction file from file {}".format(file_ID), flush = True)
            break

        # Read LocalGI's output: smooth outer pial surface        
        try:
            # smoothed pial surface            
            smoothpialv, smoothpialf = nib.freesurfer.io.read_geometry(filepath+"/surf/"+hemi+"h.pial-outer-smoothed")

            # smooth outer pial
            smoothpial_total_area = np.sum(trimesh.triangles.area(smoothpialv[smoothpialf]))
            smoothpial_volume = trimesh.triangles.mass_properties(smoothpialv[smoothpialf])['volume']

            # Transfer region thickness to smooth pial
            vlabel_smooth_pial = transfer_label(pialv, pialf, thickness, smoothpialv, smoothpialf)
            cc_smoothpial_area = partArea(smoothpialv, smoothpialf, condition = np.where((vlabel_smooth_pial==0)))
            
            # output data                
            data_dict['file_ID'] = file_ID
            data_dict['hemi'] = hemi
            data_dict['region'] = 'hemisphere'
            data_dict['total_pial_area'] = pial_total_area
            data_dict['total_pial_area_noCC'] = pial_total_area - cc_pial_area
            data_dict['total_pial_volume'] = pial_volume
            data_dict['total_smooth_pial_area'] = smoothpial_total_area
            data_dict['total_smooth_pial_area_noCC'] = smoothpial_total_area - cc_smoothpial_area
            data_dict['total_smooth_pial_volume'] = smoothpial_volume
            data_dict['total_convexhull_pial_area'] = pial_convexhull_area
            data_dict['total_convexhull_pial_area_noCC'] = pial_convexhull_area - cc_smoothpial_area
            data_dict['total_convexhull_pial_volume'] = pial_convexhull_volume
            data_dict['total_wm_area'] = white_total_area
            data_dict['total_wm_area_noCC'] = white_total_area - cc_white_area
            data_dict['total_wm_volume'] = white_volume
            data_dict['total_gm_volume'] = pial_volume - white_volume
            data_dict['thickness'] = avg_thickness
            data_dict['mean_contrast'] = mean_contrast            
            data_dict['mean_contrast_normalized'] = mean_contrast_normalized                        
            result.append(data_dict)
            
        except FileNotFoundError:            
            print("WARNING: Missing smooth outer pial from file {} hemisphere {}. Please run Freesurfer's localGI".format(file_ID, hemi), flush = True)
            smoothpial_total_area = None
            smoothpial_volume = None
            cc_smoothpial_area = None

            # output data: smooth outer pial info set to None                
            data_dict['file_ID'] = file_ID
            data_dict['hemi'] = hemi
            data_dict['region'] = 'hemisphere'
            data_dict['total_pial_area'] = pial_total_area
            data_dict['total_pial_area_noCC'] = None
            data_dict['total_pial_volume'] = pial_volume
            data_dict['total_smooth_pial_area'] = smoothpial_total_area
            data_dict['total_smooth_pial_area_noCC'] = None
            data_dict['total_smooth_pial_volume'] = smoothpial_volume
            data_dict['total_convexhull_pial_area'] = pial_convexhull_area
            data_dict['total_convexhull_pial_area_noCC'] = None
            data_dict['total_convexhull_pial_volume'] = pial_convexhull_volume
            data_dict['total_wm_area'] = white_total_area
            data_dict['total_wm_area_noCC'] = None
            data_dict['total_wm_volume'] = white_volume
            data_dict['total_gm_volume'] = pial_volume - white_volume
            data_dict['mean_contrast'] = mean_contrast            
            data_dict['mean_contrast_normalized'] = mean_contrast_normalized                        
            result.append(data_dict)
            
            continue                 
            
        except KeyboardInterrupt:
            print("The programm was terminated manually!")
            raise SystemExit                                       

    if error == False:
        file_exists = os.path.isfile(csv_output)
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
