# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:13:03 2023

@author: Matthew Yusuf
"""


import MagCrysAnisFromCIF_v1 as main
from numba import jit


cif_filename = 'Mn3Sn_OCD_1522909.cif' #filename of cif for crystal
coord_then_frac = True #True if format of symmetry operations is coordinate then fraction e.g. x+1/2, False if its otherwise e.g. 1/2+x
symop_header = '_symmetry_equiv_pos_as_xyz' #header of symmetry operations in cif file
elements_of_interest = ['Mn1'] #element of interest with '_point group symbol' appended onto it
time_inv = False #True if time inversion is included in symmetry operations and False otherwise
tolerance = None #Enter absolute tolerance to be used for comparing fractional coordinates, enter None for the precision of the most precise decimal place
#in the coordinates given in the cif file to be used
coord_sys = 'cartesian' #coordinate system of function to be entered. 'cartesian' or 'polar' for Spherical polar are the options

plot_name = "Mn3Ge xz2.html" #filename for plot
plot_title = "Mn3Ge x**2 * z**2" #title to be given to plot
strength = 1.5 #scaling factor for surface


cf = main.read_cif(cif_filename)

sym_ops = main.get_sym_ops(cf, coord_then_frac, symop_header, time_inv) #defines list of all symmetry operations
#with each element of the form (matrix, translation)

given_coordinates = main.get_coordinates(cf) #obtains fractional coordinates given in Cif file

#fix to get actual length of coordinate in cif file
if tolerance == None: #adapts absolute tolerance to precision of given coords. makes tolerance the same as the precision of the
                            #most precise given coord
    tolerance = main.get_atol(cf)

coordinates = main.coord_gen(given_coordinates,sym_ops, tolerance, time_inv)#generates all coordinates of unit cell

point_groups = main.pg_gen(coordinates, sym_ops, tolerance, time_inv) #generates point groups of every site in unit cell

eigen = main.find_eigen_sympy(elements_of_interest, point_groups) #finds eigenvectors of each point group element
#for each point group that correspond to eigenvalues of +/-1

grouped_pgs, mappings = main.group_pgs(elements_of_interest, point_groups, eigen) #groups all the point groups 
#for each element of interest by their point group symbol and collects what new atom labels the old labels
#are mapped to

grouped_elements_of_interest = list(grouped_pgs.keys()) #collects list of all the new atom labels

new_coordinates = main.group_coords(mappings, coordinates, grouped_pgs)

lattice_vector_table = main.create_lattice_vector_file(cf) #creates table of lattice vectors

element_positions, element_labels = main.get_list_from_dict_coordinates(coordinates)  #extracts element positions
#in a dictionary with keys as the element names as strings and the values as the unique element coordinates
#in the unit cell and the element labels as a list of the element names as strings
cell_coords = main.get_cell_coordinates(element_positions, cf) #generates mutiple coordinates for each element to
#be plotted

#collects positions of all the surfaces to be plotted and their orientation relative to the first site
#stores [x,y,z] coordinate positions of each atom of the desired elements and stores the 3x3 Matrix describing the orientation of the surface at each position relative
#to the original position
surface_pos, transforms = main.get_surface_pos_and_rot(grouped_elements_of_interest, new_coordinates, sym_ops, cf, lattice_vector_table, tolerance)

#set the form of the Hamiltonian term to be plotted
@jit(nopython=True)
def eng(x,y,z):
    return x**2 * z**2

main.plot_multiple_energy_surfaces_on_unit_cell_go(plot_name, plot_title, element_labels, surface_pos, transforms, eng, strength, coord_sys, *cell_coords)

