# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:03:13 2023

@author: Matthew Yusuf
"""

import pandas as pd
import sympy
import numpy as np
import scipy
import argparse

parser = argparse.ArgumentParser(prog='Create vesta file with uvw vectors from vesta file',
                    description="""This programme creates a vesta file for a unit cell, named 'vector_vesta_file_cell_name_element_of_interest', where the uvw choices
                    have been plotted at each site for the element of interest.""",
                    epilog='')

parser.add_argument('coordinates_filename', type=str, help='csv file of the fractinal coordinates of all the atoms in the unit cell.')
parser.add_argument('uvw_filename', type=str, help="""csv file that gives the point group label in HM notation, fractional coordinates and optimal uvw choices (based on symmetry axes)
for each site of the elements of interest.""")
parser.add_argument('lattice_vector_filename', type=str, help='csv file that gives the lattice vectors and lattice parameters of the unit cell.')
parser.add_argument('vesta_filename', type=str, help='vesta file of the crystal.')
parser.add_argument('element_of_interest', type=str, help='element which should have its uvw vectors plotted in the vesta file')
parser.add_argument('cell_name', type=str, help='name of the cell which will be used in the new vesta filename')
parser.add_argument('scale_factor', type=float, help='this is the factor by which the magnitude of the vectors will be multiplied. Controls how long vectors will be.')

args = parser.parse_args()
#pre-written arguments for files
#'YIG.cif_Fe1_Fe2_coordinates_v13.csv', 'YIG.cif_Fe1_Fe2_uvws_v13.csv', 'YIG.cif_Fe1_Fe2_lattice_vectors_v13.csv', 'YIG_full.vesta', 'Fe1', 'YIG', '300'
#'0.203_Mn3Ge.mcif_Mn1_1_Mn1_2_coordinates_v13.csv','0.203_Mn3Ge.mcif_Mn1_1_Mn1_2_uvws_v12.csv','0.203_Mn3Ge.mcif_Mn1_1_Mn1_2_lattice_vectors_v12.csv', '0.203_Mn3Ge_full.vesta', 'Mn1_1', 'Mn3Ge', '100'
#'0.109_Mn3Pt.mcif_Mn1_coordinates_v13.csv', '0.109_Mn3Pt.mcif_Mn1_uvws_v13.csv', '0.109_Mn3Pt.mcif_Mn1_lattice_vectors_v13.csv', '0.109_Mn3Pt.vesta', 'Mn1', 'Mn3Pt', '100'
#'Mn3Sn_OCD_1522909.cif_Mn1_coordinates_v13.csv', 'Mn3Sn_OCD_1522909.cif_Mn1_uvws_v13.csv', 'Mn3Sn_OCD_1522909.cif_Mn1_lattice_vectors_v13.csv', 'Mn3Sn_OCD_1522909.vesta', 'Mn1', 'Mn3Sn', '150'
#'0.607_RuO2.mcif_Ru1_coordinates_v13.csv', '0.607_RuO2.mcif_Ru1_uvws_v13.csv', '0.607_RuO2.mcif_Ru1_lattice_vectors_v13.csv', '0.607_RuO2.vesta', 'Ru1', 'RuO2', '100'

coordinates = pd.read_csv(args.coordinates_filename)
uvws = pd.read_csv(args.uvw_filename)
lattice_vector_table = pd.read_csv(args.lattice_vector_filename)
element_of_interest = args.element_of_interest
vector_vesta_file = open(f'vector_vesta_file_{args.cell_name}_{args.element_of_interest}_v5.vesta', 'w')
mag = 1

transform = np.zeros((3,3), dtype=float) # transfomation from fractional to cartesian to be stored here

for i in range(3):
    transform[:,i] = lattice_vector_table.iloc[i][4] * np.array(lattice_vector_table.iloc[i][1:4], dtype=float)
    
inv_transform = scipy.linalg.inv(transform) #obtains transform from cartesian to fractional


#evaluates expressions and rescales vectors
for col in uvws.columns[-9:]:
    uvws[col] = uvws[col].apply(lambda x: sympy.sympify(x).evalf() * 0.05)

old_new_indices = {}

with open(args.vesta_filename, 'r+') as f:
    f.seek(0)
    
    for line in f:
        
        if line == 'PSPGR\n':
            for pspgr in f:
                break
        
        elif line == 'GROUP\n': #sets group to P1
            vector_vesta_file.write(line)
            vector_vesta_file.write('1 1 P1\n')
            for group_line in f: #section should be only one line long so loop broken immediately
                break
            continue
        
        elif line == 'SYMOP\n': #sets sym operations as seen below
            vector_vesta_file.write(line)
            vector_vesta_file.write(' 0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1   1\n')
            vector_vesta_file.write(' -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0\n')
            
            for symop in f:
                if symop == 'TRANM 0\n': #skips past symop section 
                    vector_vesta_file.write(symop)
                    for tranm_line in f: #rewrites in case vesta file was for magnetic crystal
                        vector_vesta_file.write(' 0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1\n')
                        break
                    break
                
        elif line == 'STRUC\n':
            vector_vesta_file.write(line)
            label_map = {} #stores what element labels are mapped to which element along with column 3 value
            second_line_map = {} #stores line below fractional coordinates for each atom in vesta file
            
            for index, struc_line in enumerate(f):
                
                split_line = struc_line.split() #splits line into list
                
                if struc_line == '  0 0 0 0 0 0 0\n': #skips past struc section
                    break
                
                elif index % 2 == 0: #if line has fractional coordinates then it stores the element name and 3rd col value
                    label_map[split_line[2]] = (split_line[1], split_line[3])
                    current_label = split_line[2] #records which element label it is to map related line below to
                
                elif index % 2 == 1: #line below recorded for particular element label
                    second_line_map[current_label] = struc_line
                
                
            for index, row in coordinates.iterrows(): #loops through all coordinates and writes them into file
                
                vector_vesta_file.write(f'  {index+1} {label_map[row["atom_label"]][0]}  ' +  f'{row["atom_label"]}'.rjust(9)+f'  {label_map[row["atom_label"]][1]}   {format(float(row["x"]), ".6f")}   {format(float(row["y"]), ".6f")}   {format(float(row["z"]), ".6f")}    1         \n')
                vector_vesta_file.write(second_line_map[row["atom_label"]])
                
            vector_vesta_file.write('  0 0 0 0 0 0 0\n')
            
        elif 'THERI' in line:
            vector_vesta_file.write(line)
            theri_map = {} #maps each theri line to an element label
            
            for theri_line in f:
                
                split_line = theri_line.split() #splits line into list
                
                if theri_line == '  0 0 0\n': #skips past theri 0 section
                    break
                
                else:
                    theri_map[split_line[1]] = theri_line[theri_line.index(split_line[0])+1:] #stores line under respective atom_label key
            
            for index, row in coordinates.iterrows(): #loops through all coordinates and writes their theri 0 lines 
                vector_vesta_file.write(f'  {index+1}'+theri_map[row['atom_label']])    
            
            vector_vesta_file.write('  0 0 0\n')
        
        elif line == 'SITET\n':
            vector_vesta_file.write(line)
            sitet_map = {} #maps each sitet line to an element label
            
            for sitet_line in f:
                split_line = sitet_line.split() #splits line into list
                if sitet_line == '  0 0 0 0 0 0\n': #skips past sitet 0 section
                    break
                
                else:
                    sitet_map[split_line[1]] = sitet_line[sitet_line.index(split_line[0])+1:] #stores line under respective atom_label key
                
            for index, row in coordinates.iterrows(): #loops through all coordinates and writes their sitet 0 lines 
            
                vector_vesta_file.write(f'  {index+1}'+sitet_map[row['atom_label']])    
            
            vector_vesta_file.write('  0 0 0 0 0 0\n')
        
        elif line == 'VECTR\n':
            vector_vesta_file.write(line)
            
            vectt_index = 1
            
            for uvw_index, uvw_row in uvws.iterrows(): #loops through uvw table
                
                for coord_index, coord_row in coordinates.iterrows(): #loops through coordinate table
                    
                    #only writes vector axes for element of interest and ensures it is plotted at correct site
                    if np.all(np.isclose(np.array([uvw_row['x'], uvw_row['y'], uvw_row['z']], dtype=float), 
                                         np.array([coord_row['x'], coord_row['y'], coord_row['z']], dtype=float))) and args.element_of_interest == coord_row['atom_label']:
                        
                        #transforms uvw from cartesian to fractional
                        u = np.array(uvw_row.iloc[5:8], dtype=float)
                                    
                        u_frac = np.matmul(inv_transform, u)
                        
                        v = np.array(uvw_row.iloc[8:11], dtype=float)
                        
                        v_frac = np.matmul(inv_transform, v)
                        
                        w = np.array(uvw_row.iloc[11:14], dtype=float)
                        
                        w_frac = np.matmul(inv_transform, w)
                        
                        vector_vesta_file.write(f'{vectt_index}\t{u_frac[0]}\t{u_frac[1]}\t{u_frac[2]}\t0\n') #vector components
                        vector_vesta_file.write(f'\t{coord_index+1}\t0\t0\t0\t0\n')    #vector base position
                        vector_vesta_file.write('0 0 0 0 0 \n')
                        
                        vector_vesta_file.write(f'{vectt_index+1}\t{v_frac[0]}\t{v_frac[1]}\t{v_frac[2]}\t0\n')
                        vector_vesta_file.write(f'\t{coord_index+1}\t0\t0\t0\t0\n')
                        vector_vesta_file.write('0 0 0 0 0 \n')
                        
                        vector_vesta_file.write(f'{vectt_index+2}\t{w_frac[0]}\t{w_frac[1]}\t{w_frac[2]}\t0\n')
                        vector_vesta_file.write(f'\t{coord_index+1}\t0\t0\t0\t0\n')
                        vector_vesta_file.write('0 0 0 0 0 \n')

                
                        vectt_index += 3
            
            vector_vesta_file.write('0 0 0 0 0 \n')
            
            vector_vesta_file.write('VECTT\n')
            
            vectt_index = 1
            
            #sets magnitude and colour of each vector (u-red, v-green, w-blue)
            for index, row in uvws.iterrows():
                if element_of_interest in row['atom_type']:
                    vector_vesta_file.write(f'{vectt_index}\t{mag}\t225\t0\t0\t1\n')
                    vector_vesta_file.write(f'{vectt_index+1}\t{mag}\t0\t225\t0\t1\n')
                    vector_vesta_file.write(f'{vectt_index+2}\t{mag}\t0\t0\t225\t1\n')
                    
                    
                    vectt_index += 3
            
            vector_vesta_file.write('0 0 0 0 0 \n')
            
            #skips vectr section
            for vect_line in f:
                if vect_line == 'SPLAN\n':
                    vector_vesta_file.write(vect_line)
                    break
                   
        elif 'VECTS' in line: #writes in scaling factor
            vector_vesta_file.write(f'VECTS\t{args.scale_factor}\n')
        
        elif line == '\n':
            continue
        
        else: #writes other lines
            vector_vesta_file.write(line)

