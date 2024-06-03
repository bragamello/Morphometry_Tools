import subprocess as sb
import os
import glob
import argparse

# Parser for running
parser = argparse.ArgumentParser(prog='visualInspector_FS_meshes', description='This code produces images for visual inspection of Freesurfer meshes')
parser.add_argument('-filepath', '--file', help='path to a specific subject folder for visual inspection', required = False)
args = parser.parse_args()    

freeview_command = 'freeview -cmd {cmd} '
cmd_txt = """-v {anatomy}:grayscale=0,255 -f {lh_wm}:color=red:edgecolor=red -f {rh_wm}:color=red:edgecolor=red -f {lh_pial}:color=white:edgecolor=blue -f {rh_pial}:color=white:edgecolor=blue """  

FS_folder = args.file
cmd_file = os.path.join('./', 'cmd.txt')

sj_cmd = cmd_txt.format(anatomy=os.path.join(FS_folder, 'mri', 'orig.mgz'), lh_wm=os.path.join(FS_folder, 'surf', 'lh.white'), lh_pial=os.path.join(FS_folder, 'surf', 'lh.pial'), rh_wm=os.path.join(FS_folder, 'surf', 'rh.white'), rh_pial=os.path.join(FS_folder, 'surf', 'rh.pial'), subject='test')

with open(cmd_file, 'w') as f:
   f.write(sj_cmd)

sb.call(freeview_command.format(cmd=cmd_file), shell=True)
