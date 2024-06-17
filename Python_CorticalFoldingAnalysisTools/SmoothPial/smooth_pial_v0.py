# This function produces a smooth outer pial surface
# It is a pythonic version of the smooth pial obtained with the LocalGI Matlab script
# 
# For multiple subjects it assumes a flattened file organization
#
# derivatives
#     ---- pipeline
#         ---- subj1_ses1
#         ---- subj1_ses2
#         ---- subj2_ses1
#
# based on: A Surface-Based Approach to Quantify Local Cortical Gyrification, Marie Schaer et. al. in IEEE Transactions on Medical Imaging, vol. 27, no. 2, pp. 161-170, Feb. 2008, doi: 10.1109/TMI.2007.903576
# https://ieeexplore.ieee.org/document/4359040
#
# It has the option to save the generated surface (~5.3 MB per subject)
#
# TO DO: 1) include longitudinal file organization
#        2) Calculate the Local girification index 
#
# Victor B. B. Mello, November 2023 
# Support Center for Advanced Neuroimaging (SCAN)
# University Institute of Diagnostic and Interventional Neuroradiology
# University of Bern, Inselspital, Bern University Hospital, Bern, Switzerland.
#
# For the Newcastle+Rio+Bern (NewcarRiBe) collaboration

import argparse
import os
from multiprocessing.pool import Pool
import numpy as np
import nibabel as nib
from nibabel.processing import conform
import trimesh
from scipy import ndimage
import csv
import vtk
import vtk.util.numpy_support as np_support
from random import sample 
import time

def marchingcubes(binary, affine, filename, ns):
    # numpy to vtk array
    vtk_label_arr = np_support.numpy_to_vtk(num_array=binary.ravel(), deep=True, array_type=vtk.VTK_INT)
    vtk_img_data = vtk.vtkImageData()
    vtk_img_data.SetDimensions(binary.shape[2], binary.shape[1], binary.shape[0])
    vtk_img_data.GetPointData().SetScalars(vtk_label_arr)
    vtk_img_prod = vtk.vtkTrivialProducer()
    vtk_img_prod.SetOutput(vtk_img_data)

    # marching cubes
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(vtk_img_prod.GetOutputPort())
    surf.SetValue(0, 1)
    surf.Update()
            
    #smoothing the mesh
    smoother= vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(surf.GetOutputPort())       
    smoother.SetNumberOfIterations(int(ns)) 
    smoother.Update()
    
    # apply the affine transformation
    mat2transf = vtk.vtkMatrixToLinearTransform()
    mat = vtk.vtkMatrix4x4()
    mat.DeepCopy(affine.flatten())
    mat2transf.SetInput(mat)
    mat2transf.Update()    
    transf = vtk.vtkTransformPolyDataFilter()
    transf.SetInputConnection(smoother.GetOutputPort())
    transf.SetTransform(mat2transf)
    transf.Update()

    # Trick to get the mesh vert,faces
    # save temporary output
    # TO DO: get it from vtk.vtkTriangleFilter
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(transf.GetOutputPort())
    writer.SetFileTypeToASCII()            
    writer.SetFileName(filename)
    writer.Write()

def voxealize_mesh(mri_img, mesh, pitch=1, edge_factor=2, max_iter=10):
    # to keep in the img voxel space
    voxealized = np.zeros_like((mri_img))
    
    # remesh until all edges < max_edge (ensure vertex are inside voxel)    
    max_edge = pitch/edge_factor
    v, f, idx = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=max_iter, return_index=True)

    # convert the vertices to their voxel grid position
    hit = np.round(v / pitch).astype(int)

    # remove duplicates
    unique, _inverse = trimesh.grouping.unique_rows(hit)

    # get the voxel centers in model space
    occupied_index = np.array(hit[unique],dtype=np.int32)

    # get the voxealization
    voxealized[tuple(occupied_index.T)] = 1
    # fill the volume
    voxealized_full = ndimage.binary_fill_holes(voxealized)    

    return voxealized_full

def close_voxealization(matrix, radius, ni):
    # close the surface to obtain the smooth pial
    size = int(radius*2+3)
    x0, y0, z0 = ((size-1)/2, (size-1)/2, (size-1)/2)
    x, y, z = np.mgrid[0:size:1, 0:size:1, 0:size:1]
    r = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
    r[r >= radius] = 0
    r[r > 0] = 1
    structure=r
    closed = ndimage.binary_closing(matrix,structure=structure,iterations=ni).astype(int)
    
    return closed

def extract_smooth_pial(input_list):
    # unpack input
    filepath = input_list[0]
    args = input_list[1]    
    csv_output = args.outputcsv    
    radius = int(args.r)
    ni = int(args.ni)    
    ns = int(args.ns)    
    save_mesh = bool(args.save_pymesh)
    path2save_mesh = args.path2save_pymesh
    sample = args.sample
                    
    hemisphere = ["r","l"]
    subj = filepath.split("/")[-1]

    result = [] 
        
    for hemi in hemisphere:
        data_dict = {}
        error = False
        try:
            tic = time.time()
            # Reading FreeSurfer's reconstruction
            mri_img =  nib.load(filepath+'/mri/orig.mgz')
            pialv, pialf =  nib.freesurfer.io.read_geometry(filepath+'/surf/'+hemi+'h.pial', read_metadata=False, read_stamp=False)
            # get affine
            vox2rastkr = mri_img.header.get_vox2ras_tkr()
            rastkr2vox = np.linalg.inv(vox2rastkr)
                        
            # create a mesh of the pial surface (voxel space)
            mesh_pial = trimesh.Trimesh(nib.affines.apply_affine(rastkr2vox, pialv),pialf)
            
            # mesh voxealized            
            matrix = voxealize_mesh(mri_img.get_fdata(),mesh_pial)
                        
            # close voxalization to obtain the smooth pial
            closed = close_voxealization(matrix, radius, ni)
            
            # create pythonic mesh
            # Trick to get vertex and faces from vtk            
            if not os.path.exists('temp'):
                os.makedirs('temp')                    
            temp_filename = 'temp/{}.stl'.format(subj)
            # transpose for the vtk ordering
            marchingcubes(closed.transpose((2,1,0)),vox2rastkr,temp_filename,ns)
            py_smooth_mesh = trimesh.exchange.load.load_mesh(temp_filename)
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            vertex_defects = trimesh.curvature.vertex_defects(py_smooth_mesh)
            holes = int((4*np.pi - np.sum(vertex_defects))/4*np.pi)

            # save pythonic mesh
            if save_mesh == True:
                if not os.path.exists(path2save_mesh+'/'+subj):
                    os.makedirs(path2save_mesh+'/'+subj)            
                nib.freesurfer.io.write_geometry(path2save_mesh+'/'+subj+'/'+hemi+'h.smooth-outer-pial', py_smooth_mesh.vertices, py_smooth_mesh.faces, create_stamp=None, volume_info=None)
            tac = time.time()            
        except FileNotFoundError:
            error = True
            print("WARNING: Missing reconstruction from subject {}".format(subj), flush = True)
            break

        try:
            # Reading FreeSurfer's Smooth outer pial
            smoothpialv, smoothpialf = nib.freesurfer.io.read_geometry(filepath+"/surf/"+hemi+"h.pial-outer-smoothed")
            
            # create a smooth outer pial mesh from FS matlab script
            fs_smooth_pial = trimesh.Trimesh(smoothpialv,smoothpialf)
            data_dict['subj'] = subj
            data_dict['hemi'] = hemi
            data_dict['smooth_pial_area_python'] = py_smooth_mesh.area
            data_dict['smooth_pial_area_Matlab'] = fs_smooth_pial.area            
            data_dict['nsmooth'] = ns            
            data_dict['close_radius'] = radius
            data_dict['close_nint'] = ni
            data_dict['runtime (min)'] = (tac-tic)/60
            data_dict['holes'] = holes
                                                
            if sample:
                data_dict['sample'] = sample                                                
            
            result.append(data_dict)            

        except FileNotFoundError:
            # create a smooth outer pial mesh from FS matlab script
            print("WARNING: Missing smooth outer pial from subject {} hemisphere {}. Needs to run Freesurfer's localGI".format(subj, hemi), flush = True)
            data_dict['subj'] = subj
            data_dict['hemi'] = hemi
            data_dict['smooth_pial_area_python'] = py_smooth_mesh.area
            data_dict['smooth_pial_area_Matlab'] = None            
            data_dict['nsmooth'] = ns            
            data_dict['close_radius'] = radius
            data_dict['close_nint'] = ni
            data_dict['runtime (min)'] = (tac-tic)/60
            data_dict['holes'] = holes
                                    
            if sample:
                data_dict['sample'] = sample                                                            

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

# Parser for shell script
parser = argparse.ArgumentParser(prog='smooth_pial_v0', description='This code computes the smooth pial from a Freesurfer reconstruction file in a hemisphere base for one or multiple subjects. Mutiple subjects assumes a flattened file organization')
parser.add_argument('-filepath', '--path', help='path to the folder containing the Freesurfer reconstructions for one or multiple subjects', required = True)
parser.add_argument('-outputfile', '--outputcsv', default = 'SmoothPial.csv', help='Path to the csv output file. Default is SmoothPial.csv to be saved at the running directory', required = False)
parser.add_argument('-nsmooth', '--ns', default = int(500), help='Smoothing steps for the created mesh', required = False)
parser.add_argument('-radius', '--r', default = int(7), help='Radius used to close the surface', required = False)
parser.add_argument('-nint', '--ni', default = int(4), help='number of interactions to closed the smooth pial', required = False)
parser.add_argument('-save_mesh', '--save_pymesh', default = False, help='Save the smooth pial mesh', required = False)
parser.add_argument('-path_save_mesh', '--path2save_pymesh', default = 'temp', help='Path to save the smooth pial mesh. Tobe used together with -save_mesh True. The temp file created is the default path', required = False)
parser.add_argument('-sample_name', '--sample', help='Name for the sample processed', required = False)
args = parser.parse_args()

fs_files_path = args.path
check_list = sorted(list(os.listdir(args.path)))

# check if it is a single subject
if 'mri' in check_list and 'surf' in check_list:
    input_list = (fs_files_path,args)
    extract_smooth_pial(input_list)

# Multiple subjects
else:
    subj_list = check_list
    input_list = [(fs_files_path +'/'+i, args) for i in subj_list]

    # proccess multiple subjects at the same time
    with Pool() as pool:
        res = pool.map(extract_smooth_pial, input_list)
