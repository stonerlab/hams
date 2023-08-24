# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:02:09 2023

@author: Matthew Yusuf
"""

import argparse
import MagCrysAnisFromCIF_v1 as main


parser = argparse.ArgumentParser(prog='Find UVW for sites on unit cell',
                    description="""This programme takes a .cif or .mcif file as input and outputs csv files with the following information:
                        
                        CIF_filename_elements_of_interest_uvws_v15.csv - gives the point group label in HM notation, fractional coordinates and optimal uvw choices (based on symmetry axes)
                        for each site of the elements of interest.
                        
                        CIF_filename_elements_of_interest_wv_xyz_v15.csv - gives the point group elements from which the symmetry axes were chosen for w and also v if
                        another symmetry axis was found in (x,y,z) form. returns the identity for both point group elements if there were no symmetry axes. Also returns the 
                        order of the point group elements and the point group label in HM notation.
                        
                        CIF_filename_elements_of_interest_lattice_vectors_v15.csv - gives the lattice vectors and lattice parameters of the unit cell.
                        
                        CIF_filename_elements_of_interest_property_tensor_file_v15.csv - returns the elements of the even rank magnetocrystalline anisotropy property tensors 
                        up to the highest rank (taken as input) in a table format. Returns each set of tensors for each site of the elements of interest.
                        
                        CIF_filename_elements_of_interest_coordinates_v15.csv - returns a table of the fractional coordinates of each atom in the unit cell.
                        """,
                    epilog='')

parser.add_argument('filename', type = str, help="""This is the filename of the .cif or .mcif file of the crystal structure to be analysed. Should contain a list of symmetry 
                    operations and lattice parameters and lattice angles, along with the coordinates for one site of each element""")

parser.add_argument('symop_header', type = str, help='The header of the symmetry operations given in the .cif or .mcif file')
parser.add_argument('elements_of_interest', type = str, nargs = '+', help='These are the names of the elements for which the tables should be produced.')

parser.add_argument('--highest_rank', type = int, default = 6, help="""This is the highest rank of magnetocrystalline anisotropy property tensor to be considered. This input 
                    must be an even integer. This input is not required and by default it is 6""")
                    
parser.add_argument('--time_inv', choices = ['true', 'false', 'True', 'False', 'T', 'F'], required=True, help="""This specifies whether or not the symmetry operations in the
                    .cif or .mcif file include a time reversal symmetry operation. This argument is required.""")

parser.add_argument('--sym_op_form', choices = ['coord_then_frac', 'frac_then_coord'], required=True, help="""Specifies whether the symmetry operations are written in the format
                    of coordinate then translation or translation then coordinate. e.g. coordinates then translation would be x+1/2 whilst the other would be 1/2+x.
                    This argument is required.""")

parser.add_argument('--tolerance', type = float, default=None, help="""This is the tolerance value for the comparison of fractional coordinates when comparing them in the 
                    analysis. This input is not required and by default the tolerance will be the same as the order of the most precise decimal place found on any coordinate.""")



args = parser.parse_args()
#pre-written arguments for files
#'0.607_RuO2.mcif', '_space_group_symop_magn_operation.xyz', 'Ru1', '--time_inv', 'true', '--sym_op_form', 'coord_then_frac'
#'Mn3Sn_OCD_1522909.cif', '_symmetry_equiv_pos_as_xyz', 'Mn1', '--time_inv', 'False', '--sym_op_form', 'coord_then_frac'
#'0.203_Mn3Ge.mcif', '_space_group_symop_magn_operation.xyz', 'Mn1_1', 'Mn1_2', '--time_inv', 'true', '--sym_op_form', 'coord_then_frac'
#'0.109_Mn3Pt.mcif', '_space_group_symop_magn_operation.xyz', 'Mn1', '--time_inv', 'true', '--sym_op_form', 'coord_then_frac'
#'YIG.cif', '_symmetry_equiv_pos_as_xyz', 'Fe1', 'Fe2', '--time_inv', 'false', '--sym_op_form', 'coord_then_frac'
#'PbO2.cif', '_symmetry_equiv_pos_as_xyz', 'Pb1', '--time_inv', 'F', '--sym_op_form', 'frac_then_coord'


cif_filename = args.filename #enter Cif filename here

highest_rank = args.highest_rank #enter the rank of the highest rank tensor you want to be included in calculation


#enter True if time symmetry operations are included in the CIF file
if args.time_inv == 'true' or args.time_inv == 'True' or args.time_inv == 'T':
    time_inv = True

elif args.time_inv == 'false' or args.time_inv == 'False' or args.time_inv == 'F' :
    time_inv = False 


#True if the format of the symmetry operations in the CIF are of the format
#x+1/3, y+1/3, z+1/3 and False if the format is 1/3+x, 1/3+y, 1/3+y
if args.sym_op_form == 'coord_then_frac':
    coord_then_frac = True 

elif args.sym_op_form == 'frac_then_coord':
    coord_then_frac = False

symop_header = args.symop_header # enter the name of the header for the symmetry operations in 
#the CIF file as a string


elements_of_interest = args.elements_of_interest #enter desired atom types as strings 


if highest_rank%2 != 0:
    raise ValueError("highest rank tensor must be of even rank; odd rank tensors are null for magneteocrystalline anisotropy")

cf = main.read_cif(cif_filename) #reads in data from Cif file

sym_ops = main.get_sym_ops(cf, coord_then_frac, symop_header, time_inv) #defines list of all symmetry operations
#with each element of the form (matrix, translation)

given_coordinates = main.get_coordinates(cf) #obtains fractional coordinates given in Cif file

#fix to get actual length of coordinate in cif file
if args.tolerance == None: #adapts absolute tolerance to precision of given coords. makes tolerance the same as the precision of the
                            #most precise given coord
    tolerance = main.get_atol(cf)
else:
    tolerance = args.tolerance


coordinates = main.coord_gen(given_coordinates,sym_ops, tolerance, time_inv) #generates all coordinates of unit cell

point_groups = main.pg_gen(coordinates, sym_ops, tolerance, time_inv) #generates point groups of every site in unit cell

eigen = main.find_eigen_sympy(elements_of_interest, point_groups) #finds eigenvectors of each point group element
#for each point group that correspond to eigenvalues of +/-1

grouped_pgs, mappings = main.group_pgs(elements_of_interest, point_groups, eigen) #groups all the point groups 
#for each element of interest by their point group symbol and collects what new atom labels the old labels
#are mapped to

#unique_pgs = main.get_unique_pg_indices(grouped_pgs) #groups indices of sites with the same point grouop that have
#the same orientation

grouped_elements_of_interest = list(grouped_pgs.keys()) #collects list of all the new atom labels

new_coordinates = main.group_coords(mappings, coordinates, grouped_pgs)

lattice_vectors_table = main.create_lattice_vector_file(cf) #creates a table of lattice vectors and lattice
#parameters

grouped_eigen = main.eigen_frac2cart(main.find_eigen_sympy(grouped_elements_of_interest, grouped_pgs), lattice_vectors_table) #finds dictionary of eigenvalues
#and eigenvectors grouped by the new atom labels. eigenvectors are converted to cartesian basis

orders = main.find_orders(grouped_elements_of_interest, grouped_pgs) #finds the orders of the point group 
#elements for the first point group of each desired atom type

axes_0, wv_choices = main.find_first_axes(grouped_elements_of_interest, orders, grouped_eigen) #finds basis for first site of each 
#atom type using symmetry axes found from the eigenvectors of the point group elements

wv_pges = {} #stores the point group operations from which w and v are taken

for element in grouped_elements_of_interest:
    wv_pges[element] = main.wv_pge_finder([pge[0] for pge in grouped_pgs[element][0]], wv_choices[element])

xyz_wv_table = main.xyz_wv_pges(wv_pges) #creates table of x,y,z forms of symmetry axes used to produce w and 
#v axes
    
axes = main.find_all_axes(grouped_elements_of_interest, sym_ops, new_coordinates, axes_0, lattice_vectors_table, tolerance) #uses the symmetry 
#operations to transform the axes from the first site to all of the sites

uvw_table = main.create_uvw(grouped_elements_of_interest, new_coordinates, axes) #creates a table of the uvw values 
#for each atom with the atom's fractional coordinates and atom type given

props_dict = main.prop_tensor_dict(grouped_elements_of_interest, grouped_pgs, highest_rank) #creates dictionary of property tensors for each atom_type for each
#site

#First_Hamiltonian = main.create_ham_file(grouped_elements_of_interest, grouped_pgs, highest_rank, time_inv) #stores
#Hamiltonian for first site of each atom type in a table

#Hamiltonians = main.get_atom_hamiltonians(grouped_elements_of_interest, new_coordinates, grouped_pgs, highest_rank, time_inv) 
#stores Hamiltonians of all sites for each element of interest

coord_table = main.make_coord_table(coordinates) #stores all coordinates for unit cell for use in vesta file maker programme

#saves tables to files
element_string = ''
for element in elements_of_interest:
    element_string += element + '_'
    
filename_prefix = f'{args.filename}_{element_string}'

xyz_wv_table.to_csv(f'{filename_prefix}wv_xyz_v15.csv', index=False)

uvw_table.to_csv(f'{filename_prefix}uvws_v15.csv', index=False)

#First_Hamiltonian.to_csv(f'{filename_prefix}single_Ham_v15.csv', index=False)

#Hamiltonians.to_csv(f'{filename_prefix}all_Ham_v15.csv', index=False)

lattice_vectors_table.to_csv(f'{filename_prefix}lattice_vectors_v15.csv', index=False)

main.write_prop_tensors_file(props_dict, new_coordinates, f'{filename_prefix}_property_tensor_file_v15')

coord_table.to_csv(f'{filename_prefix}coordinates_v15.csv', index=False)