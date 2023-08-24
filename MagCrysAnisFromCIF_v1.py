# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:50:07 2023

@author: Matthew Yusuf
"""

import numpy as np
import re
from CifFile import ReadCif
from fractions import Fraction
import sympy
import opt_einsum
import string

#for plots, from  numba import jit for plotting 
import plotly.graph_objs as go
from itertools import product

#for tables
import pandas as pd


def read_cif(filename):
    """Returns dictionary like object containing Cif file information.
    
    Reads in data from cif file using ReadCIf function from CifFile module and ensures that the correct
    object is stored in the variable as the necessary information may be nested in one key."""

    cf_file = ReadCif(filename)
    if len(cf_file.keys()) == 1:
        return cf_file[cf_file.keys()[0]]
    
    return cf_file
    

def symop_from_quadruplet(quadruplet, time_inv=True):
    """Returns a symmetry operation from a string based triplet as used in cif formats
    
    :param quadruplet: string quadruplet representing a symmetry operation (e.g. 'y+3/4, x+1/4, -z+x-y+1/4, -1')
    :return: 3x3 symmetry operation rotation, 3x1 symmetry operation translation, integer time symmetry operation
    quadruplet can also be triplet representing a symmetry operation (e.g. 'y+3/4, x+1/4, -z+x-y+1/4')
    where the time symmetry operation is not returned
    """
    
    # Split the quadruplet string into the four componets and each component is further
    # split into a 'sign', 'letter', 'letter2', 'letter3', 'fraction' and 'time inversion' part (e.g. '-', 'z', '+x', '-y', '1/4','-1').
    tokenized_triplet = re.findall(r'([\+\-])?([xyz])([\+\-][xyz])?([\+\-][xyz])?([\+\-][0-9]/[0-9])?',quadruplet)

    if len(tokenized_triplet) != 3:
        raise ValueError('triplet must have three parts')
        
    # Simple map to change x,y,z into matrix and vector indices
    mapping = {'x': 0, 'y': 1, 'z': 2}
    
    matrix = np.zeros((3, 3), dtype=int)
    trans = np.zeros(3)
    
    for row, row_data in enumerate(tokenized_triplet):
        sign, letter, letter2, letter3, frac = row_data
        
        # Sign can be '+', '-', or '' (i.e. positive with the '+' omitted) so we
        # write the sign as a string before converting to a float
        col = mapping[letter]
        matrix[row, col] = int(f'{sign}1')
        
        #repeats this process for the other two letters if they are present
        if letter2:
            col2 = mapping[list(letter2)[1]]
            matrix[row, col2] = int(f'{list(letter2)[0]}1')
            
        if letter3:
            col3 = mapping[list(letter3)[1]]
            matrix[row, col3] = int(f'{list(letter2)[0]}1')
        
        # Use the fraction module to convert frac (a string) into a float. If
        # the frac string is empty then the translation is 0.
        trans[row] = float(Fraction(frac)) if frac else 0.0
        
    if time_inv: #will return time inversion symmetry if it is present
        return sympy.Matrix(matrix), sympy.Matrix(trans), int(quadruplet[-2:])
    return sympy.Matrix(matrix), sympy.Matrix(trans)


def shift_fractional_coordinate_to_zero_one(coord):
    """Returns an array based on coord where all values are in the range [0; 1)"""
    for n in range(len(coord)):
        if coord[n] < 0.0:
            coord[n] = coord[n] - np.floor(coord[n])
        if coord[n] > 1.0:
            coord[n] = coord[n] - np.floor(coord[n])
        if np.isclose(float(coord[n]), 1.0): #did float because sympy float type not supported
            coord[n] = 0.0
    return coord


def symop_from_triplet_alt(triplet, time_inv=True):
    """Returns a symmetry operation from a string based triplet as used in cif formats
    
    :param triplet: string triplet representing a symmetry operation (e.g. '3/4+y, 1/4+x, 1/4-z+x-y')
    
    :return: 3x3 symmetry operation rotation, 3x1 symmetry operation translation
    
    triplet can also be quadruplet representing a symmetry operation (e.g. '3/4+y, 1/4+x, 1/4-z+x-y', -1)
    
    where an extra term in the tuple is returned, the time inversion
    """
    
    # Split the triplet string into the three componets and each component is further
    # split into a 'fraction', 'sign' and 'letter'  part (e.g. '1/4', '-', 'z', '+','x','-', 'y' ).
    tokenized_triplet = re.findall(r'([\+\-]?[0-9]/[0-9])?([\+\-])?([xyz])([\+\-])?([xyz])?([\+\-])?([xyz])?', triplet)
    if len(tokenized_triplet) != 3:
        raise ValueError('triplet must have three parts')
        
    # Simple map to change x,y,z into matrix and vector indices
    mapping = {'x': 0, 'y': 1, 'z': 2}
    
    matrix = np.zeros((3, 3),dtype=int)
    trans = np.zeros(3)
    
    for row, row_data in enumerate(tokenized_triplet):
        frac, sign, letter, sign2, letter2, sign3, letter3 = row_data
        
        # Sign can be '+', '-', or '' (i.e. positive with the '+' omitted) so we
        # write the sign as a string before converting to a float
        col = mapping[letter]
        matrix[row, col] = int(f'{sign}1')
        
        #repeats this process for the other two letters if they are present
        if letter2:
            col2 = mapping[letter2]
            matrix[row, col2] = int(f'{sign2}1')
        if letter3:
            col3 = mapping[letter3]
            matrix[row, col3] = int(f'{sign3}1')
        
        # Use the fraction module to convert frac (a string) into a float. If
        # the frac string is empty then the translation is 0.
        trans[row] = float(Fraction(frac)) if frac else 0.0
        
    if time_inv: #will return time inversion symmetry if it is present
        return sympy.Matrix(matrix), sympy.Matrix(trans), int(triplet[-2:])
    return sympy.Matrix(matrix), sympy.Matrix(trans)


def get_sym_ops(cf_dict, coordthenfrac, key = "_space_group_symop_magn_operation.xyz", time_inv = True):
    """Returns a list of symmetry operations from the cf_dict object, where each symmetry 
    operation is of the form (rot, trans, inv). 
    
    coordthenfrac should be True when the symmetry is written as coordinates then fraction, and false when
    it is written in reverse order. Causes unwanted no translations to be extracted from CIF file if
    specified incorrectly.
    
    Key specificies where in the cif file to find the symmetry operations. set to 
    "_space_group_symop_magn_operation.xyz" however for non-magnetic crystals the key is commonly
    "_space_group_symop_operation_xyz"
    
    time_inv specifies whether or not time inversion is included in the symmetry operations.
    """

    if coordthenfrac:
        return [symop_from_quadruplet(i, time_inv) for i in cf_dict[key]]
        
    return [symop_from_triplet_alt(i, time_inv) for i in cf_dict[key]]


def get_coordinates(cf_dict, key = "_atom_site_label"):
    """Returns fractional coordinates given in cf_dict as a dictionary containing 
    the labels of each element as keys and the fractional coordinates in the Cif file as a 3x1 sympy 
    Matrix object for the respective values. 
    
    Key specifies where in the cif file to find the fractional coordinates of each element. Set to 
    "_atom_site_label" as default.

    Removes any uncertainty in the coordinates, usually shown by a bracket at the end of the coordinates.
    """
    
    given_coordinates = {} #creates dictionary to store coordinates for each element
    j=0
    
    
    for element in cf_dict[key]:#loops through each element
        #returns coordinates as 3x1 sympy
        given_coordinates[element] = sympy.Matrix([float(re.findall(r'[-+]?(?:\d*\.*\d+)', cf_dict[f'_atom_site_fract_{i}'][j])[0]) for i in ['x','y','z'] ])
        j+=1
        if len(given_coordinates[element]) != 3:
            raise ValueError("fractional coordinates must have three parts")
            
        #shifts coords inside unit cell
        given_coordinates[element] = shift_fractional_coordinate_to_zero_one(given_coordinates[element])
        
    return given_coordinates


def coord_gen(cif_coords, sym_ops, atol, time_inv = True):
    """Returns a nested dictionary for the coordinates in the unit cell with the main dictionary having the
    elements as the keys and the nested dictionary containing integer key values and each coordinate 
    represented by a 3x1 sympy Matrix stored as a value 
    
    Takes cif_coords, a dictionary with elements as keys and as elements a list of the fractional 
    coordinates from the cif file in the form of a 3x1 sympy Matrix, sym_ops a list of symmetry operations 
    from the cif file and atol which is the absolute tolerance to be used for comparing fractional coordinates 
    in np.isclose. If time_inv = True then each element of sym_ops is assumed to be of the form 
    (rotation, translation, time inversion), and if False it is taken to be of the form 
    (rotation, translation).
    
    Coordinates calculated by applying symmetry operations to fractional coordinates
    given from the Cif file and each new coordinate is added to the coordinates dictionary with its 
    corresponding symmetry operation added to the point_groups dictionary.
    """
    #creates dictionary coordinates with each element as keys
    coordinates = {}
    
    #variable to track when a new coordinate is found and when a new key needs to be made for it in each
    #dictionary
    i=0  
    
    #boolean variable which changes once the coordinate is found to be the same as an exisiting one in the 
    #dictionaries and moves iteration onto next symmetry operation
    break_again = False
    
    
    for element in cif_coords:
        
        if len(cif_coords[element])!=3:
            raise ValueError("fractional coordinates must have three parts")
            
        for coord in cif_coords[element]:
            if coord > 1.0 or coord < 0.0:
                raise ValueError("coordinates must be on interval [0,1)")
        
        coordinates[element] = {i:cif_coords[element]} #creates dictionary relating each coordinate to an integer
        
    for element in cif_coords: #loop through elements
        i=0
        for sym in sym_ops: #loop through sym ops
            
            if sym[0].shape != (3,3):
                raise ValueError("rotation matrix must be of dimension (3,3)")
            if sym[1].shape != (3,1):
                raise ValueError("translation matrix must be of dimension (3,1)")
            if time_inv:
                if sym[2] != 1 and sym[2] != -1:
                    raise ValueError("time inversion must be integer of value -1 or 1")
            
            break_again = False
            #calculates new coordinate using symmetry operation
            new_coord = shift_fractional_coordinate_to_zero_one(sym[0]*cif_coords[element] + sym[1])
            
            #loops through existing coordinates to check if coordinate already found
            for index in coordinates[element]:
                if np.all(np.isclose(np.array(new_coord, dtype=float),
                                         np.array(coordinates[element][index], dtype=float), atol = atol)):
                    #if coordinate found then loop is broken
                    break_again =True
            if break_again:
                continue
            
            #if coordinate not found it is added to list
            i+=1
            coordinates[element][i] = new_coord
            
    return coordinates

def pg_gen(coordinates, sym_ops, atol, time_inv = True):
    """Returns a nested dictionary for the point groups
    each position with the main key being the element and the nested dictionary containing the point groups
    as values where their key is the same key as the coordinate they represent in the coordinates dictionary.
    
    Variables
    
    coordinates - a nested dictionary for the coordinates in the unit cell with the main dictionary having the
    elements as the keys and the nested dictionary containing integer key values and each coordinate 
    represented by a 3x1 sympy Matrix stored as a value.
    
    sym_ops - list of symmetry operations from the cif file 
    
    atol - the absolute tolerance to be used for comparing fractional coordinates in np.isclose
    
    time_inv - if True then each element of sym_ops is assumed to be of the form 
    (rotation, translation, time inversion), and if False it is taken to be of the form 
    (rotation, translation).
    
    ---
    
    point groups calculated by applying all symmetry operations to each site and whichever operations leave the 
    coordinates invariant has their rotation added to the point group of that site.
    """
    point_groups = {} #creates dicitonary for point groups
    for element in coordinates: #loops through elements
        point_groups[element] = {} #sets each element as a key
        #loops through all coordinates
        for coordinate_index in coordinates[element]: #loops through coordinates of each site for each element
            old_coord = coordinates[element][coordinate_index] 
            point_groups[element][coordinate_index] = [] #sets each coordinate index as a sub key for a point group list value
            #loops through all symmetry operations
            for sym in sym_ops:
                #calculates new coordinate using symmetry operation
                new_coord = shift_fractional_coordinate_to_zero_one(sym[0]*old_coord + sym[1])
                
                #checks if coordinate is invariant under symmetry operation
                if np.all(np.isclose(np.array(new_coord, dtype=float),
                                         np.array(old_coord, dtype=float),
                                         atol=atol)) and sym not in point_groups[element][coordinate_index]:
                    

                    #appends rotation from symmetry operation to point group and time inversion
                    #factor if time inversion is considered
                    if time_inv:
                        point_groups[element][coordinate_index].append((sym[0],sym[2]))
                    else:
                        point_groups[element][coordinate_index].append((sym[0],))
            print(f"point group {coordinate_index+1} done of {len(coordinates[element])} for {element}")

    return point_groups
    

def get_unique_pg_indices(pgs):
    """Returns indices of unique point group sites as a dictionary where the keys are the elements and the 
    values are arrays containing the indices of the sites which have the same point group, oriented in the 
    same way.
    
    Takes pgs which is a nested dictionary with elements as keys and as nested keys takes the integers 
    associated with certain coordinates in the coordinates dictionary generated from the pg_and_coord_gen
    function with the values being lists of the point group operations for each coordinate.
    """

    unique_pgs = {}
    for element in pgs:
        unique_pgs[element]=[]
        for coordinate in pgs[element]:
            unique_pgs[element].append([])

        for index1 in pgs[element]:
            for index2 in pgs[element]:
                if type(pgs[element][index1]) != type(pgs[element][index2]):
                    raise ValueError("Elements of each list must be of same type")
                if pgs[element][index1] == pgs[element][index2]:
                    unique_pgs[element][index1].append(index2)
        unique_pgs[element] = list(np.unique(np.array(unique_pgs[element]),axis=0))
        
    return unique_pgs


def generate_hamiltonian_cart(prop_tensors):
    """Returns the hamiltonian in Cartesian coordinates for the site of the crystal based on its magnetocrystalline anisotropy 
    property tensors
    
    Accepts list of property tensors.
    """
    
    #creates symbols for each direction cosine
    symbols = sympy.symbols('alpha_1:4') 
    x,y,z = symbols
    dircos = sympy.Array([x,y,z],3)
    
    #creates anisotropy value to be returned
    anis = 0
    
    #used to create indices for contraction
    letters = dict((key[0], key[1]) for key in enumerate(string.ascii_lowercase))
    
    potential_b_terms = []
    
    for prop_tensor in prop_tensors: #loops through each given property tensor
        rank = len(prop_tensor.shape) #determines the rank of the tensor
    
        #based on rank a tensor product with the necessary number of direction cosines is done followed by
        #a tensor contraction over the necessary indices. This is then added to the running total anisotropy 
        #value
        indices = ''
        for i in range(rank):
            indices += letters[i] #creates indices of property tensor
    
        for j in range(rank):
            indices += ',' + letters[j] #creates indices of each direction cosine tensor
        
        indices += '->'
        
        #tensor contraction
        anis += opt_einsum.contract(indices,prop_tensor, *[dircos for i in range(rank)])
        
        dim_string = "(1:4)"
        
        potential_b_terms += sympy.symbols(f'b{dim_string*rank}') #creates list of all possible b terms
    
    b_poly_form = anis.as_poly(potential_b_terms) #expresses anisotropy as a polynomial all possible b terms
    #to find all the combinations of the alpha1, alpha2 and alpha3 terms
    
    anis = b_poly_form.as_expr() #converts polynomial to sympy expression
    
    xyz_term_dict = b_poly_form.as_dict() #creates dictionary with the values being the coefficients of the 
    #b terms which are expressions of alpha1, alpha2 and alpha3
    
    xyz_term_temp_assign = [] #temporarily reassigns each expression of alpha1, alpha2 and alpha3 so that b terms can
    #be collected and reassigned
    labels = [] #stores labels for temporary reassignment
    counter = 0
    
    for xyz in xyz_term_dict: #loops through possible terms of alpha1, alpha2 and alpha3
        A = sympy.symbols(f'A{counter}') 
        xyz_term_temp_assign.append((xyz_term_dict[xyz],A)) #stores tuple of alpha1, alpha2 and alpha3 term and 
        #temporary reassignment variable
        labels.append(A) #stores temp reassignment variables
        counter += 1
        
    anis = anis.subs(xyz_term_temp_assign) #substitutes temp variables
    
    A_poly_form = anis.as_poly(labels) #expresses hamiltonian as a polynomial of temp variables
    
    anis = A_poly_form.as_expr() #converts to sympy expression
    
    b_term_dict = A_poly_form.as_dict() #creates dictionary with values being grouped b terms
    
    reassignments = [] #stores tuples of old coefficients and new K value to be substituted in its place
    counter = 0 #produces new K value for each coefficient
    K_terms = [] #stores new K values created
    for term in b_term_dict: #iterates through all coefficients 
        K = sympy.symbols(f'K{counter}') #creates new K value
        reassignments.append((b_term_dict[term],K)) #adds tuple of old and new coefficients to list
        K_terms.append(K) #adds new K terms to list
        counter += 1
    
    anis = anis.subs(reassignments, simultaneous=True) #subs in new coefficients
    
    #resubstitutes original expressions for alpha1, alpha2 and alpha3 and removes temp variables
    anis = anis.subs([(temp_sub[1], temp_sub[0]) for temp_sub in xyz_term_temp_assign], simultaneous=True)
    
    anis = anis.expand()
    
    #identities for alpha1, alpha2 and alpha3 to be subbed in
    #up to 6th order identities included
    
    identities = []
    s_expr = x**2*y**2 + x**2*z**2 + y**2*z**2
    p_expr = x**2 * y**2 * z**2
    s, p = sympy.symbols('s, p')
    Id1 = x**6 + y**6 + z**6
    Id2 = (x**4 * y**2 + x**2 * y**4) + (x**4 * z**2 + x**2 * z**4) + (y**4 * z**2 + y**2 * z**4)
    Id3 = x**4 + y**4 + z**4
    Id4 = x**2 + y**2 + z**2
    Id_LHS = [Id1, Id2, Id3, Id4]

    #adds each identity to a identities list as a tuple of the form (LHS, RHS) 
    identities.append((Id1, 1 - 3*s +3*p))
    identities.append((Id2, s - 3*p))
    identities.append((Id3,1-2*s))
    identities.append((Id4,1)) # trying diff
    identities.append((p_expr,p))
    identities.append((s_expr,s))
    
    w = sympy.Wild('w', exclude=symbols)
    v = sympy.Wild('v',exclude=symbols)
    
    collected = False
    
    #collects the common coefficient terms
    anis = sympy.collect(anis.together(),[w*K for K in K_terms]+[(w*K)/v for K in K_terms]+K_terms) 
    
    while not collected: #loops until all identities have been subbed in as much as possible
        
        anis = anis.subs(identities).expand() #subs in identities
        
        #collects the common coefficient terms
        anis = sympy.collect(anis.together(),[w*K for K in K_terms]+[(w*K)/v for K in K_terms]+K_terms) 
        
        for Id in Id_LHS: #checks if any more identities can be subbed in
            if str(Id) not in str(anis):
                
                collected = True #if no more can be subbed in then loop is broken
    
    anis = sympy.collect(anis.expand(),[s,p]) #collects terms as coefficients of s and p
    
    new_K_subs = [] #collects K values for new groups of K coefficients
    
    new_K_terms = []
    
    new_poly_form_dict = anis.as_poly(s,p).as_dict() #dictionary of coefficients
    
    for new_term in new_poly_form_dict: #loops through coefficients
        
        #checks if alpha1, alpha2 and alpha3 is in coefficient to ensure these terms are not removed from expression
        if str(x) not in str(new_poly_form_dict[new_term]) and str(y) not in str(new_poly_form_dict[new_term]) and str(z) not in str(new_poly_form_dict[new_term]):
            
            new_K = sympy.symbols(f'K{counter}')
            
            #adds old coefficients of k terms and new k terms to list as tuple to be used for substitution
            new_K_subs.append((new_poly_form_dict[new_term], new_K))
            new_K_terms.append(new_K)
            counter += 1
    
    #subs in new K terms and original expressions for s and p
    anis = anis.subs([(p, p_expr), (s, s_expr)]+new_K_subs, simultaneous=True)
    
    #collects the common coefficient terms
    anis = sympy.collect(anis.expand(), K_terms+new_K_terms) 
    
    const_expr = 0 #created to store a collection of standalone K coefficients
    
    for term in anis.make_args(anis.expand()): #loops through arguments in expression
        if str(x) not in str(term) and str(y) not in str(term) and str(z) not in str(term): #checks if term
            #is constant
            const_expr += term #adds constant terms to expression
        
    #subs in one constant for group of constants and collects common K terms
    return sympy.collect(anis.expand().subs(const_expr, sympy.symbols(f'K{counter}')), K_terms+new_K_terms)


def CartesianToSpherical(expr):
    """Takes sympy expression in Cartesian coordinates (S_x,S_y,S_z) and converts it to spherical polar coordinates"""
    x,y,z = sympy.symbols('S_x:z')
    theta = sympy.symbols('theta')
    phi = sympy.symbols('phi')
    alpha1 = sympy.sin(theta)*sympy.cos(phi)
    alpha2 = sympy.sin(theta)*sympy.sin(phi)
    alpha3 = sympy.cos(theta)
    
    return sympy.simplify(expr.subs([(x, alpha1), (y, alpha2), (z, alpha3)]))


#found online at https://notebook.community/adriaanvuik/solid_state_physics/crystal_lattice

def unit_cell_3d(a, b, c, atom_pos, Nx, Ny, Nz):
    """Make arrays of x-, y- and z-positions of a lattice from the
    lattice vectors, the atom positions and the number of unit cells.
    
    Parameters:
    -----------
    a : list
        First lattice vector
    b : list
        Second lattice vector
    c : list
        Third lattice vector
    atom_pos : list
        Positions of atoms in the unit cells in terms of a, b and c
    Nx : int
        number of unit cells in the x-direction and negative x direction to be plotted
    Ny : int
        number of unit cells in the y-direction and negative y direction to be plotted
    Nz : int
        number of unit cells in the z-direction and negative z direction to be plotted
        
    Returns:
    --------
    latt_coord_x : numpy.ndarray
        Array containing the x-coordinates of all atoms to be plotted
    latt_coord_y : numpy.ndarray
        Array containing the y-coordinates of all atoms to be plotted
    latt_coord_z : numpy.ndarray
        Array containing the z-coordinates of all atoms to be plotted
    """
    latt_coord_x = []
    latt_coord_y = []
    latt_coord_z = []
    for atom in atom_pos:
        xpos = atom[0]*a[0] + atom[1]*b[0] + atom[2]*c[0]
        ypos = atom[0]*a[1] + atom[1]*b[1] + atom[2]*c[1]
        zpos = atom[0]*a[2] + atom[1]*b[2] + atom[2]*c[2]
                
        xpos_all = [xpos + n*a[0] + m*b[0] + k*c[0] for n, m, k in
                     product(range(-Nx, Nx), range(-Ny, Ny), range(-Nz, Nz))]
        ypos_all = [ypos + n*a[1] + m*b[1] + k*c[1] for n, m, k in
                     product(range(-Nx, Nx), range(-Ny, Ny), range(-Nz, Nz))]
        zpos_all = [zpos + n*a[2] + m*b[2] + k*c[2] for n, m, k in
                     product(range(-Nx, Nx), range(-Ny, Ny), range(-Nz, Nz))]
        latt_coord_x.append(xpos_all)
        latt_coord_y.append(ypos_all)
        latt_coord_z.append(zpos_all)
    latt_coord_x = np.array(latt_coord_x).flatten()
    latt_coord_y = np.array(latt_coord_y).flatten()
    latt_coord_z = np.array(latt_coord_z).flatten()
    return np.array([latt_coord_x, latt_coord_y, latt_coord_z])


def get_list_from_dict_coordinates(coordinates):
    """takes dictionary with keys being elements and values being nested dictionaries with integers as keys
    and lists of coordinates as values of the form [x,y,z] and returns dictionary with elements as keys
    and coordinates as values and a list of the element labels as strings"""
    element_positions = {}
    element_labels = []
    for element in coordinates:
        element_positions[element] = []
        element_labels.append(element)
        for position in coordinates[element]:

            element_positions[element].append(list(np.array(list(coordinates[element][position]),dtype=float)))
    return element_positions, element_labels


def get_cell_coordinates(element_positions, cf):
    """returns a list of arrays of each elements positions with the arrays being of the form of output of the
    unit_cell_3d function, and the index of the list corresponds to the index of the element in element_labels
    output from the get_list_from_dict_coordinates function"""
    
    #extracting lattice parameters
    cell_lengths = [] #stores cell lengths a,b,c
    cell_angles = [] #stores angles alpha, beta, gamma
    for cell_len in ['a','b','c']:
        cell_lengths.append(float(re.findall(r'[-+]?(?:\d*\.*\d+)', cf[f'_cell_length_{cell_len}'])[0]))
    for cell_angle in ['alpha', 'beta', 'gamma']:
        cell_angles.append(float(re.findall(r'[-+]?(?:\d*\.*\d+)', cf[f'_cell_angle_{cell_angle}'])[0]))

    #stores cell length and angle values
    len_a = cell_lengths[0]
    len_b = cell_lengths[1]
    len_c = cell_lengths[2]
    #convert all angles to radians for numpy functions
    alpha = np.deg2rad(cell_angles[0])
    beta = np.deg2rad(cell_angles[1])
    gamma = np.deg2rad(cell_angles[2])
    #normalises all vectors with respect to 1
    b_norm = len_b/len_a
    c_norm = len_c/len_a

    sins = [] #stores the sin values of each angle
    coss = [] #stores the cos values of each angle

    #stores the cos and sin value of each angle and does checks to ensure the small numbers close to 0 that 
    #numpy outputs are set to 0
    for angle in [alpha,beta,gamma]:
        sin = np.sin(angle)
        cos = np.cos(angle)

        if np.isclose(sin,0):
            sin = 0
        if np.isclose(cos,0):
            cos=0
        sins.append(sin)
        coss.append(cos)

    #general lattice vectors
    #a and b set to be in the same plane and c is determined from its relations with a and b
    a = [1,0,0]
    b = [b_norm*coss[2], b_norm*sins[2], 0]
    c = [c_norm * coss[1], c_norm * (coss[0] - (coss[2]*coss[1]))/(len_b * sins[2]),
        c_norm * np.sqrt((sins[1])**2 - (((coss[0] - (coss[1]) * coss[2])**2)/(len_b**2 * (sins[2])**2)))]
    
    #atom positions
    cell_coords = []
    for element in element_positions:
        cell_coords.append(unit_cell_3d(a,b,c, element_positions[element], 1, 1, 1))

    return cell_coords


def plot_multiple_energy_surfaces_on_unit_cell_go(filename,plot_title,element_labels, surface_locs, transforms, energy, strength=1.0, coords='polar', *args):
    """outputs html file containing an interactive plot of the cell with the energy density surfaces for each
    position plotted at the respective positions in their respective orientation.
    
    filename specifies the name of the file and filepath to where the file should be stored, as string in which the plot is to be stored.
    
    plot_title is a string which is used as the title of the plot
    
    element labels is a list of strings with the names of the elements being plotted in the same order
    in which the elements are entered into the *args argument.
    
    surface_locs is a list of length 3 list type objects containing the x,y,z coordinates at which each
    energy density surface is to be plotted.
    
    transforms is a list of 2D array like objects which can be cast to numpy array which descibe the 
    of the surfaces at each site. The indices of surface_locs and transforms are related and each describe
    the same site.
    
    energy is a function which takes input as integers, floats or arrays of such and returns a single scalar 
    or an array of scalar values depending on input. this function takes either three values (x,y,z) if 
    coords = 'cartesian' or 2 values (theta, phi) if coords = 'polar'.
    
    strength is a float value which is the scaling factor for the surface to be plotted.
    
    each argument in *args is a list of arrays containing arrays of dimension 3xN where N is the number of
    positions of the atom and each of the first 3 dimensions describes the x,y,z position of the atom.
    """
    #################################################
    # VARIABLE

    # Function sampling
    # Smaller --> faster
    # Bigger --> prettier
    num_points = 500

    # Create a list of thetas
    T = np.linspace(0, np.pi, num_points, dtype=np.float64)

    # Create a list of phis
    P = np.linspace(0, 2*np.pi, num_points, dtype=np.float64)

    thetas, phis = np.meshgrid(T, P)

    if coords=='polar':
        radius = strength*energy(thetas, phis)

    elif coords=='cartesian':

        xs = np.sin(thetas)*np.cos(phis)
        ys = np.sin(thetas)*np.sin(phis)
        zs = np.cos(thetas)

        radius = strength*energy(xs, ys, zs)


    else:
        raise Exception('Coordinates must be specified. `polar` and `cartesian` are accepted')

    radius = radius - np.min(radius)

    X = radius * np.sin(thetas) * np.cos(phis)
    Y = radius * np.sin(thetas) * np.sin(phis)
    Z = radius * np.cos(thetas)    

    xyz = np.array([X,Y,Z]) #stores all meshgrid coordinates

    fig = go.Figure()
    
    loc_counter = 0 #tracks which surface is being plotted
    
    #plots energy density surface with orientation based on transforms matrices and position based on
    #surface_locs vectors
    for loc in surface_locs:
        
        shape = xyz.shape
        
        rot_xyz = np.zeros(shape, dtype=float) #stores rotated coordinates
        #loops through meshgrid coordinates
        for index1 in range(shape[1]): 
            for index2 in range(shape[2]):
                #rotates all coordinates by transformation matrix
                rot_xyz[:, index1, index2] = np.array(transforms[loc_counter] * sympy.Matrix(xyz[:, index1, index2]), dtype=float).reshape((3,))
        
        #plots each surface at desired position
        fig.add_surface(x=(rot_xyz[0,:,:] + float(loc[0])),
                        y=(rot_xyz[1,:,:] + float(loc[1])),
                        z=(rot_xyz[2,:,:] + float(loc[2])), opacity=0.75, showscale=False)
        
        loc_counter+=1
        print(f'Surface {loc_counter} done of {len(surface_locs)}')
    
    element_counter = 0
    marker_symbols = ['circle', 'diamond', 'square', 'circle-open',
            'diamond-open', 'square-open', 'cross', 'x']
    symbol_counter = 0
    
    #plots elements in cell
    for arg in args:
        fig.add_scatter3d(x=arg[0],y=arg[1],z=arg[2], mode='markers', marker_symbol=marker_symbols[symbol_counter], name=element_labels[element_counter])
        element_counter += 1
        symbol_counter += 1
    fig.update_layout(
    scene = dict(
        xaxis = dict(range=[-0.25,1],),
                     yaxis = dict(range=[-0.25,1],),
                     zaxis = dict(range=[-0.25,1],),),
    title = dict(text=plot_title))
    
    fig.write_html(filename)
    

def get_surface_pos_and_rot(elements_of_interest, coordinates, sym_ops, cf, lattice_vector_table, atol):
    """returns tuple of list of [x,y,z] coordinate positions for the surfaces and list of 3x3 sympy Matrices 
    describing the orientation for the surfaces
    
    Variables: 
    
    -elements_of_interest: list which describes the desired elements on which a surface is to be placed
    
    -coordinates: a nested dictionary with the strings of element names as keys and position index
    of each atom of that element as a sub key and the fractional coordinates of the atom position as values
    
    -sym_ops: sym_ops - list of symmetry operations from the cif file 
    
    -cf: the dictionary like object that is output from the read_cif function containing all of the 
    information from the CIF file
    
    -lattice_vector_table: Pandas DataFrame object lattice_vectors which contains the lattice vectors and 
    lattice parameters of the cell.
    
    -atol: the absolute tolerance to be used for comparing fractional coordinates in np.isclose
    """
    
    #retrieves cell lengths from CIF file
    cell_lengths = []
    for cell_len in ['a','b','c']:
        cell_lengths.append(float(re.findall(r'[-+]?(?:\d*\.*\d+)', cf[f'_cell_length_{cell_len}'])[0]))
    
    a = cell_lengths[0] #lattice parameter a
    
    transform = get_transform_from_lattice_vectors(lattice_vector_table) #transforms fractional to cartesian coordinates
    
    inv_transform = transform.inv() #transforms cartesian to fractional coordinates
    
    surface_pos = [] #stores positions of surface as positions of the element of interest and converts these coordinates to
    #normalised coordinates
    rots = [] #stores the rotation of each position wrt to original position by taking them from relative point
    #groups
    
    for EOI in elements_of_interest: #loops through list of desired elements
        first_coord = coordinates[EOI][0] #stores first coordinate
        for position in coordinates[EOI]: #loops through coordinates of unit cell
            coordinate = coordinates[EOI][position] #extracts coordinates of position
            for symop in sym_ops: #loops through symmetry operations
                new_coord = shift_fractional_coordinate_to_zero_one((symop[0] * first_coord) + symop[1]) #finds transformed coordinate
                
                #if transformed coordinate is the current coordinate then the rotation used in the 
                #transformation is stored and used as orientation of surface
                if np.all(np.isclose(np.array(new_coord, dtype=float), 
                                     np.array(coordinate, dtype=float), atol=atol)):
                    rots.append(transform*symop[0]*inv_transform) #changes rotation to cartesian basis
                    break
            
            #stores coordinates of position in the list of surface position in cartesian coordinates and converts the units to the
            #units normalised by a
            surface_pos.append(transform * coordinate * (1/a))
    return surface_pos, rots


def get_atom_hamiltonians(elements_of_interest, coordinates, point_groups, highest_rank, time_inv, 
                              polar = False):
    """
    returns a pandas datafame with the columns 'atom_type','x','y','z','Ham', where x,y,z columns are 
    fractional coordinates and Ham contains the hamiltonian of the atom at that particular fractional 
    coordinate. Each row represents a different atom
    
    Variables
    
    elements of interest is a list of element labels as strings which are to be added to the output dictionary
    
    coordinates is a nested dictionary with the strings of atom_type as keys and position index
    of each atom of that atom_type as a sub key and the fractional coordinates of the atom position as values
    
    point_groups is a nested dictionary with the strings of atom_type as keys and position index
    of each atom of that atom_type as a sub key and the point groups of the atom_type at that site as values
    where the point groups are a list of 3x3 sympy matrices in a tuple with a time inversion (1 or -1). 
    Note that the tuple only contains one element if time_inv is false.
    
    highest_rank specifies the highest rank of magnetocrystalline anisotropy tensor which should be included
    in analysis. Must be an even number for mangetocrystalline anisotropy.
    
    time_inv is True if time inversion is included in the CIF file and False otherwise
    
    polar is True if the hamiltonian terms need to be output in polar coordinates, and false otherwise.
    """
    
    if highest_rank%2 != 0:
        raise ValueError("highest rank tensor must be of even rank; odd rank tensors are null for magneteocrystalline anisotropy")
    
    num_rows = sum([len(coordinates[element]) for element in elements_of_interest])
    counter = 0
    coord_ham_table = pd.DataFrame(columns=['atom_type','x','y','z','Ham'])
    
    #do property tensors instead 
    
    for element in elements_of_interest: #loops through desired elements
        
        for coordinate_index in coordinates[element]:
            coord = coordinates[element][coordinate_index]
            Ham = generate_hamiltonian_cart([crystal_tensor(i, *[pge[0] for pge in point_groups[element][coordinate_index]])
                                             for i in range(2, highest_rank + 1, 2)])
            coord_ham_table.loc[len(coord_ham_table)] = [element, coord[0], coord[1], coord[2], Ham]
            counter+=1
            print(f'hamitonian {counter} done of {num_rows}')
    return coord_ham_table


def create_ham_file(elements_of_interest, point_groups, highest_rank, time_inv, polar = False):
    """returns table with columns of 'atom_type' and 'Ham' where each row stores the atom_type with the 
    corresponding Hamiltonian for the first site of that atom_type as a sympy expression.
    
    variables
    
    elements_of_interest - a list of element labels as strings
    
    point_groups - a nested dictionary with the strings of atom_type as keys and position index
    of each atom of that atom_type as a sub key and the point groups of the atom_type at that site as values
    where the point groups are a list of 3x3 sympy matrices in a tuple with a time inversion (1 or -1). 
    Note that the tuple only contains one element if time_inv is false.
    
    highest_rank - highest rank of property tensor to be considered in calculation
    
    time_inv - True if time inversion operation included in point group, false if not
    
    polar - True if hamitonian is to be returned in spherical polar coordinates and false if it is to be 
    returned in Cartesian
    """
    Hamiltonians = pd.DataFrame(columns=['atom_type','Ham'])
    
    for atom_type in elements_of_interest:
        if not polar:
            Hamiltonians.loc[len(Hamiltonians)] = [atom_type,
                                              generate_hamiltonian_cart([crystal_tensor(i, *[pge[0] for pge in point_groups[atom_type][0]]) 
                                                                         for i in range(2, highest_rank + 1, 2)])]
        if polar:
            Hamiltonians.loc[len(Hamiltonians)] = [atom_type, CartesianToSpherical(generate_hamiltonian_cart([crystal_tensor(i, *[pge[0] for pge in point_groups[atom_type][0]]) 
                                       for i in range(2, highest_rank + 1, 2)]))]
    
    return Hamiltonians


def create_lattice_vector_file(cf):
    """returns table with lattice vectors and parameters. takes in cf, a dictionary like object that is 
    output from the read_cif function containing all of the information from the CIF file
    """
    #extracting lattice parameters
    cell_lengths = [] #stores cell lengths a,b,c
    cell_angles = [] #stores angles alpha, beta, gamma
    for cell_len in ['a','b','c']:
        cell_lengths.append(float(re.findall(r'[-+]?(?:\d*\.*\d+)', cf[f'_cell_length_{cell_len}'])[0]))
    for cell_angle in ['alpha', 'beta', 'gamma']:
        cell_angles.append(float(re.findall(r'[-+]?(?:\d*\.*\d+)', cf[f'_cell_angle_{cell_angle}'])[0]))

    #stores cell length and angle values
    len_b = cell_lengths[1]
    #convert all angles to radians for numpy functions
    alpha = np.deg2rad(cell_angles[0])
    beta = np.deg2rad(cell_angles[1])
    gamma = np.deg2rad(cell_angles[2])

    sins = [] #stores the sin values of each angle
    coss = [] #stores the cos values of each angle

    #stores the cos and sin value of each angle and does checks to ensure the small numbers close to 0 that 
    #numpy outputs are set to 0
    for angle in [alpha,beta,gamma]:
        sin = np.sin(angle)
        cos = np.cos(angle)

        if np.isclose(sin,0):
            sin = 0
        if np.isclose(cos,0):
            cos=0
        sins.append(sin)
        coss.append(cos)

    #general lattice vectors
    #a and b set to be in the same plane and c is determined from its relations with a and b
    a = [1,0,0]
    b = [coss[2], sins[2], 0]
    c = [coss[1], (coss[0] - (coss[2]*coss[1]))/(len_b * sins[2]),
        np.sqrt((sins[1])**2 - (((coss[0] - (coss[1] * coss[2]))**2)/(len_b**2 * (sins[2])**2)))]

    lattice_vectors = pd.DataFrame(columns=['Lattice_vector','x','y','z','Lattice_parameter'])
    
    #stores the lattice vectors and parameters in the table
    index = 1
    for vector in [a,b,c]:
        lattice_vectors.loc[len(lattice_vectors)] = [f'e_{index}',float(vector[0]),float(vector[1]),
                                                     float(vector[2]), float(cell_lengths[index-1])]
        index+=1
    return lattice_vectors


def create_uvw(elements_of_interest, coordinates, axes):
    """returns pandas DataFrame with columns atom_type', 'point group symbol', 'x','y','z','u_x','u_y',
    'u_z','v_x','v_y','v_z','w_x','w_y','w_z' where x,y,z are the fractional coordinates of each atom, and
    each of u,v,w are x,y,z coordinates of the Cartesian basis vectors of each atom's orientation 
    expressed in the basis of the shortest Hamitonian.
    
    Variables
    
    elements_of_interest - a list of element labels as strings
    
    coordinates - a nested dictionary with the strings of atom_type as keys and position index
    of each atom of that atom_type as a sub key and the fractional coordinates of the atom position as 
    values
    
    axes - a nested dicitonary with element labels as keys and atom site indices as sub keys with a list
    of 3 3x1 sympy Matrices representing the uvw axes as values in cartesian coordinates.
    """
    
    uvw_table = pd.DataFrame(columns=['atom_type', 'point group symbol','x','y','z','u_x','u_y','u_z','v_x','v_y','v_z',
                                      'w_x','w_y','w_z'])
    
    for atom_type in elements_of_interest:
        reverse_atom_type_list = list(atom_type)
        
        reverse_atom_type_list.reverse() #reverses list of atom_type string characters so that first 
        #occurring _ is the one before the appended pg symbol
        
        _index = len(atom_type) - reverse_atom_type_list.index('_') #obtains starting index of point group 
        #symbol
        
        pg_symbol = atom_type[_index:] #finds pg symbol as it should be after 
        #underscore in atom_type string
        
        for coordinate_index in coordinates[atom_type]:
            coord = coordinates[atom_type][coordinate_index]
            site_axes = axes[atom_type][coordinate_index]
            uvw_table.loc[len(uvw_table)] = [atom_type, pg_symbol, coord[0], coord[1], coord[2],
                                            site_axes[0][0], site_axes[0][1], site_axes[0][2],
                                            site_axes[1][0], site_axes[1][1], site_axes[1][2],
                                            site_axes[2][0], site_axes[2][1], site_axes[2][2]]
    
    for index, row in uvw_table.iterrows():
        uvw_table.iloc[index,2:] = row[2:].apply(lambda x: 0.0 if np.isclose(float(x),0) else float(x)) 
         
    return uvw_table


#for checking group axioms

def is_mat_equal(A, B):
    """checks if two matrices A and B are equal"""
    
    return np.all(np.isclose(np.array(A,dtype=float),np.array(B,dtype=float)))

def find_identity(pg):
    """finds identity of given point group"""
    
    found = False
    for possible_id in pg:
        counter =0
        for pge in pg:
            
            if is_mat_equal(possible_id * pge, pge) and is_mat_equal(pge * possible_id, pge):
                
                counter += 1
        
        if counter == len(pg):
            if found:
                raise ValueError('Two identities found')
            identity = possible_id
            found = True
                
    if found:
        return identity
    raise ValueError("point group has no identity")

def contains_inv(pg):
    """returns true if array contains inverse for each element"""
    for pge in pg:
        if pge.inv() not in pg:
            return False
    return True

def is_closed(pg):
    """returns true is point group is closed"""
    for pge1 in pg:
        for pge2 in pg:
            counter = 0
            for pge_check in pg:
                if not is_mat_equal(pge_check, pge1 * pge2):
                    counter +=1
            if counter == len(pg):
                return False
    return True

#---

def crystal_tensor(rank, *args):
    """Generates a magnetocrystalline anisotropy property tensor of specified rank for a set of 3x3 
    Matrices for a point group. Returns a sympy Array. Only works for Polar I property tensors."""

    dim = []
    letters = dict((key[0], key[1]) for key in enumerate(string.ascii_lowercase))
    
    if rank%2 == 1 or type(rank) != int or rank == 0:
        raise ValueError('rank must  be even positive integer.')
    
    #generates appropriate indices for tensor contraction
    indices = ''
    for i in range(rank):
        indices += letters[2*i] + letters[2*i+1] +','
    for j in range(rank):
        indices += letters[2*j+1]
    indices += '->'
    for k in range(rank):
        indices += letters[2*k]

    for i in range(rank):
        dim.append(3)
        
    onetofour = '(1:4)'
    b_old = sympy.Array(sympy.symbols(f'b{onetofour*rank}'),tuple(dim))
    
    for mat in args: #loops through generator matrices
        
    
        rot = []
        for i in range(rank):
            rot.append(mat)
        
        #performs tensor contraction
        b_new = opt_einsum.contract(indices, *rot, b_old)

        equations = [] 

        #reshapes tensor so that each element contains one expression
        #this makes it easier to loop through
        for equation in (b_new-b_old).reshape(1,3**rank)[0]:
            equations.append(equation)

        #solves all the equations obtained and substitutes into the property tensor
        solutions = sympy.solve(equations, sympy.symbols(f'b{onetofour*rank}'))

        b_old = b_old.subs(solutions) #subs solutions into original matrix
    
    return b_old


def find_eigen_sympy(elements_of_interest, point_groups, keep_complex=False):
    """returns dictionary with elements_of_interest members as string keys and point group indices as
    sub keys (with reference to the index of the point group in point groups) and indices of the point
    group element as sub sub keys with a list with a list of the real eigenvalues that are +/-1 in the 
    first index and a list of the corresponding eigenvectors at the second index.
    
    variables:
    
    elements_of_interest is a list of string labels ofor the desired elements.
    
    point_groups: a nested dictionary with the strings of atom_type as keys and position index
    of each atom of that atom_type as a sub key and the point groups of the atom_type at that site as values
    where the point groups are a list of 3x3 sympy matrices in a tuple with a time inversion (1 or -1). 
    Note that the tuple only contains one element if time_inv is false.
    """
    
    #using sympy and keeping only real eigenvalues +/-1
    #keeps all eigenvalues for each matrix in one list
    eigen = {}

    for element in elements_of_interest:
        eigen[element] = {} #creates sub dictionary for each element

        for pg_index in point_groups[element]:
            eigen[element][pg_index] = {} #creates sub dictionary for each point group
            
            #generates list of the eigenvalue and eigenvector pairs for each point group element
            pg_eigen_list = [pge[0].eigenvects() for pge in point_groups[element][pg_index]]

            index = 0
            
            #loops through lists of tuples for each pge
            for pge in pg_eigen_list:
                eigen[element][pg_index][index] = []
                
                #loops through list of eigenvalue, multiplicity, eigenvector tuples for each pg element
                for eigenvalue, multiplicity, eigenvectors in pge:
                    
                    if not keep_complex:
                    
                        #only adds real eigenvalues that are +/-1
                        if eigenvalue == 1 or eigenvalue == -1:
                            for eigenvector in eigenvectors:
                                eigen[element][pg_index][index].append((eigenvalue, eigenvector))
                    else:
                        for eigenvector in eigenvectors:
                                eigen[element][pg_index][index].append((eigenvalue, eigenvector))
                index += 1
    return eigen


def find_orders(elements_of_interest, point_groups):
    """returns dictionary with elements of interest as string keys and (point group element index, element 
    order) tuples as values for the first point group of the desired elements
    
    variables:
    
    elements_of_interest is a list of string labels ofor the desired elements
    
    point_groups: a nested dictionary with the strings of atom_type as keys and position index
    of each atom of that atom_type as a sub key and the point groups of the atom_type at that site as values
    where the point groups are a list of 3x3 sympy matrices in a tuple with a time inversion (1 or -1). 
    Note that the tuple only contains one element if time_inv is false."""
    
    identity = sympy.Matrix([[1,0,0],
                        [0,1,0],
                        [0,0,1]])
    
    orders = {}
    
    for element in elements_of_interest: #loops through desired elements
        orders[element] = [] #creates list for each element
        index = 0 #tracks which point group element is being analysed
        
        for pge in point_groups[element][0]:
            order_not_found = True 

            order = 1 #tracks current poewr of group element
            product = identity #sets group element to identity

            while order_not_found: #takes powers of group element until it equals identity

                product *= pge[0] 

                if is_mat_equal(product, identity): #checks if power of group element equals identity
                    orders[element].append((index, order)) #appends group element index and order to list
                    order_not_found = False #breaks loop once order found

                order += 1
            index += 1
    return orders


def find_first_axes(elements_of_interest, orders, eigen):
    """finding axes on each site by performing symmetry operations on axes of first site
    stores three axes in list [u,v,w] where u will be the highest symmetry axis, and v will be an orthogonal
    symmetry axis if there exists one. w is then u x v. also returns list of point group elements from 
    which u and v came, in a tuple with their order.
    
    Variables
    
    elements_of_interest - a list of string labels ofor the desired elements
    
    eigen - a nested dictionary with element labels as keys and point group indices as sub keys and point
    group element index as sub sub key and a list of the (eigenvalue, eigenvector) tuples in cartesian 
    coordinates for that point group element as values
    
    orders - a dictionary with element label as key and tuples of the form (pg element index, order) as 
    values where the pg in question is the pg of the first site
    
    ---
    
    Process for choosing axes -
    
    chooses high sym axis as w, if high sym axis is a mirror or 2 fold rotation then the axis of rotation or
    axis perpendicular to the plane is chosen as w, and then a low sym axis is searched for. if no 
    orthogonal low sym axes exist then the high sym axis is checked for orthogonal real eigenvectors,
    followed by the low sym axis if none are found from the high sym axis. 
    
    If these do not exist then arbitrary vectors are chosen from the plane orthogonal to the high sym axis, 
    where 0 and 1 are substituted into the equation for the plane into the two independent variables to 
    obtain vectors in the plane. Inversion axes are ignored, and if the highest symmetry is an inversion
    then the axes are taken to be the usual cartesian axes ([1,0,0], [0,1,0], [0,0,1]).
    """

    axes_0 = {}
    wv_choices = {} #stores tuples of (pge index, order) for w and v in that order

    for element in elements_of_interest: #loops through desired elements
        wv_choices[element] = []
        
        sorted_order = sorted(orders[element], key = lambda pair: pair[1], reverse=True) #sorts
        #group elements by order in descending order 

        high_sym_eigen = eigen[element][0][sorted_order[0][0]] #takes highest order element's eigenvalue and
        #eigenvector as the high symmetry axis

        high_sym_plus = []

        high_sym_minus = []

        axes_0[element] = []

        for eigen_element in high_sym_eigen:
            if eigen_element[0] == 1: #stores eigenvectors with eigenvalue +1 in plus
                high_sym_plus.append(eigen_element)
            elif eigen_element[0] == -1: #stores eigenvectors with eigenvalue -1 in plus
                high_sym_minus.append(eigen_element)

        #only scenario there is more than 1 eigenvector in the high symmetry axis is if its a refelection 
        #or a 2 fold rotation or an inversion and this will have three orthogonal eigenvectors, hence the third one of the 
        #different sign will be obtained from the cross product later on 
        
        axes_0[element] = []
        
        if len(high_sym_minus) == 1: #picks high sym axis
            axes_0[element].append((high_sym_minus[0][0] * high_sym_minus[0][1]).normalized())
            wv_choices[element].append(sorted_order[0]) #w choice

        elif len(high_sym_plus) == 1: #picks high sym axis
            axes_0[element].append((high_sym_plus[0][0] * high_sym_plus[0][1]).normalized())
            wv_choices[element].append(sorted_order[0]) #w choice
        
        if len(axes_0[element]) == 0: #if high sym is not rot, inv rot or mirror plane (-1 or 1 pg)
            axes_0[element] = [sympy.Matrix([1,0,0]), sympy.Matrix([0,1,0]), sympy.Matrix([0,0,1])]
        
        low_axis_found = False #breaks loop if low sym axis found
        
        for pge_index, order in sorted_order[1:]:
            low_sym_eigen = eigen[element][0][pge_index] #stores eigenvalues and eigenvectors of current pge
            
            low_sym_plus = []

            low_sym_minus = []
        
            for eigen_element in low_sym_eigen:
                if eigen_element[0] == 1: #stores eigenvectors with eigenvalue +1 in plus
                    low_sym_plus.append(eigen_element)
                elif eigen_element[0] == -1: #stores eigenvectors with eigenvalue -1 in plus
                    low_sym_minus.append(eigen_element)
            
            if len(low_sym_minus) == 1: #picks low sym axis
                potential_axis = (low_sym_minus[0][0] * low_sym_minus[0][1]).normalized()
                if axes_0[element][0].dot(potential_axis) == 0:
                    axes_0[element].append(potential_axis)
                    wv_choices[element].append((pge_index, order)) #v choice
                    low_axis_found = True

            elif len(low_sym_plus) == 1: #picks low sym axis
                potential_axis = (low_sym_plus[0][0] * low_sym_plus[0][1]).normalized()
                if axes_0[element][0].dot(potential_axis) == 0:
                    axes_0[element].append(potential_axis)
                    wv_choices[element].append((pge_index, order)) #v choice
                    low_axis_found = True
            
            if low_axis_found:
                break
            
        if len(axes_0[element]) == 1: #all other sym axes linearly dependent on high sym axis
            
            #check if high sym has orthogonal eigenvectors
            if len(high_sym_minus) == 2:
                potential_axis = (high_sym_minus[0][0] * high_sym_minus[0][1]).normalized()
                
                if axes_0[element][0].dot(potential_axis) == 0:
                    axes_0[element].append(potential_axis) 
                    wv_choices[element].append(sorted_order[0]) #v choice
            
            elif len(high_sym_plus) == 2:
                potential_axis = (high_sym_plus[0][0] * high_sym_plus[0][1]).normalized()
                
                if axes_0[element][0].dot(potential_axis) == 0:
                    axes_0[element].append(potential_axis)
                    wv_choices[element].append(sorted_order[0]) #v choice
                    
        if len(axes_0[element]) == 1: #if high sym axis had no orthogonal eigenvectors check low sym
            
            #check if low sym has orthogonal eigenvectors
            if len(low_sym_minus) == 2:
                potential_axis = (low_sym_minus[0][0] * low_sym_minus[0][1]).normalized()
                
                if axes_0[element][0].dot(potential_axis) == 0:
                    axes_0[element].append(potential_axis)
                    wv_choices[element].append((pge_index, order)) #v choice
            
            elif len(low_sym_plus) == 2:
                potential_axis = (low_sym_plus[0][0] * low_sym_plus[0][1]).normalized()
                
                if axes_0[element][0].dot(potential_axis) == 0:
                    axes_0[element].append(potential_axis)
                    wv_choices[element].append((pge_index, order)) #v choice
        
        if len(axes_0[element]) == 1: #if high sym axis had no orthogonal eigenvectors vectors chosen 
            #arbitrarily from plane perpendicular to high sym axis
            
            general_vect = sympy.Matrix(sympy.symbols('a:c'))
            general_vect = general_vect.subs(sympy.solve(sympy.Eq(axes_0[element][0].dot(general_vect),0), 
                                                         dict=True)[0])
            sub_list = []
            sub_val = 0
            for coord in general_vect:
                if str(coord) == 'a' or str(coord) == 'b' or str(coord) == 'c':
                    sub_list.append((coord,sub_val))
                    sub_val += 1
            general_vect = general_vect.subs(sub_list)
            axes_0[element].append(general_vect.normalized())

        #Once two axes obtained, third is produced from cross product. high sym axis is taken to be w
        #low sym axis is taken to be v. u = v x w
        
        if len(axes_0[element]) == 2:
            axes_0[element].append((axes_0[element][1].cross(axes_0[element][0])).normalized())
        
        rearranged_axes = [axes_0[element][2], axes_0[element][1], axes_0[element][0]]
        
        axes_0[element] = rearranged_axes
        
    return axes_0, wv_choices
        

def find_all_axes(elements_of_interest, sym_ops, coordinates, axes_0, lattice_vector_table, atol):
    """generating axes for other sites from axes of first site
    
    variables
    
    elements of interest - list of string labels of desired elements
    
    sym_ops - list of symmetry operations from CIF file as (rotation matrix, translation) and possibly
    includes time inversion if that is given in the CIF but not required
    
    coordinates - dictionary with keys being elements and values being nested dictionaries with integers as keys
    and lists of coordinates as values of the form [x,y,z] and returns dictionary with elements as keys
    and coordinates as values and a list of the element labels as strings
    
    axes_0 - dicitonary with elements as keys and list of 3 sympy 3x1 Matrices as values representing the 
    u,v,w axes in cartesain coordinates.
    
    lattice_vector_table - a pandas DataFrame with columns of Lattice_vector, x, y, z and Lattice_parameter which are used to 
    convert the fractional coordinates to cartesian
    
    atol - the absolute tolerance to be used for comparing fractional coordinates in np.isclose
    """
    
    transform = get_transform_from_lattice_vectors(lattice_vector_table) #transforms fractional coordinates to cartesian
    
    inv_transform = transform.inv() #inverse is used to transform cartesian to fractional
    
    axes = {} 
    for element in elements_of_interest: #loops through desired elements
        axes[element] = {0: axes_0[element]} #adds axes of first site from axes_0 input

        for symop in sym_ops: #loops through all symmetry operations
            
            #calculates new coordinate
            new_coord = shift_fractional_coordinate_to_zero_one(symop[0]*coordinates[element][0] + symop[1])
            
            #loops through existing coordinates to find which coordinate was found
            for coordinate_index in coordinates[element]:
                
                if np.all(np.isclose(np.array(new_coord, dtype=float), 
                                     np.array(coordinates[element][coordinate_index], dtype=float), atol=atol)):
                    new_coordinate_index = coordinate_index #captures coordinate index of calculated coordinate
    
                    if new_coordinate_index not in axes[element]: #checks if coordinate axes has already been calculated
                            
                        new_axes = []
                        for axis in axes_0[element]: #transforms each axis from the first site to desired site
                            new_axes.append((transform * (symop[0] * inv_transform * axis)).normalized()) #transforms to fractional to be rotated then transformed back to cart
                        axes[element][new_coordinate_index] = new_axes #adds transformed axes to dictionary 
    return axes


def find_pg_orders(pg):
    """returns dictionary with point group element index as a key and the order of the element as a value
    
    variable
    
    pg - list of point group operations as 3x3 matrices"""
    
    identity = sympy.Matrix([[1,0,0],
                        [0,1,0],
                        [0,0,1]])
    
    pg_orders = {}
    
    index = 0
    for pge in pg: #loops through pg elements
        order_not_found = True 

        order = 1 #tracks current poewr of group element
        product = identity #sets group element to identity

        while order_not_found: #takes powers of group element until it equals identity

            product *= pge 

            if is_mat_equal(product, identity): #checks if power of group element equals identity
                pg_orders[index] = order #appends group element index and order to list
                order_not_found = False #breaks loop once order found

            order += 1
        
        
        index +=1
    return pg_orders


def find_pg_symbol(pg, eigen, pg_orders):
    """finds the Hermann–Mauguin symbol for the point group passed into function for any of the 32 
    crystallographic classes
    
    variables
    
    point_group - list of point group operations as 3x3 matrices
    
    eigen - dictionary with pge index as key and as values a list of (eigenvalue, eigenvector) 
    tuples for each point group element
    
    pg_orders - a dictionary with the point group element index as a key and the order of the element as a 
    value"""
    
    pg_order = len(pg)
    
    if pg_order == 1:
        return '1'
    
    if pg_order == 16:
        return '4/mmm'
    
    if pg_order == 3:
        return '3'
    
    if pg_order == 48:
        return 'm-3m'
    
    if pg_order == 2:
        
        pos_det_counter = 0 #tracks number of pure rotations
        
        index = 0
        
        for pge in pg:
            negative_eigenvalue_counter = 0 #tracks number of negative eigenvalues
            positive_eigenvalue_counter = 0 #tracks number of positive eigenvalues
            
            if pge.det() == 1:
                pos_det_counter +=1 #increments by one for each pure rotation
            
            for eigen_tuple in eigen[index]:
                if eigen_tuple[0] == -1:
                    negative_eigenvalue_counter +=1 #negative eigenvalues
                if eigen_tuple[0] == 1: #positive eigenvalues
                    positive_eigenvalue_counter +=1
            
            if negative_eigenvalue_counter == 3:
                return '-1'

            if negative_eigenvalue_counter == 1 and positive_eigenvalue_counter == 2:
                return 'm'
            
            index += 1
            
        if pos_det_counter == 2: #if all are pure rotation
            return '2'

    if pg_order == 4:
        pos_det_counter = 0 #tracks number of pure rotations
        order2_counter = 0 #tracks number of order 2 elements
        order4_counter = 0 #tracks number of order 4 elements
        mirror_counter = 0 #tracks number of pure mirrors
        index = 0
        
        for pge in pg:
            
            negative_eigenvalue_counter = 0 #tracks number of negative eigenvalues
            positive_eigenvalue_counter = 0 #tracks number of positive eigenvalues
            
            if pge.det() == 1:
                pos_det_counter +=1 #increments by one for each pure rotation
            if pg_orders[index] == 2:
                order2_counter += 1 #increments by one for each order 2 element
            if pg_orders[index] == 4:
                order4_counter += 1 #increments by one for each order 4 element
            
            for eigen_tuple in eigen[index]:
                if eigen_tuple[0] == -1:
                    negative_eigenvalue_counter +=1 #negative eigenvalues
                if eigen_tuple[0] == 1:
                    positive_eigenvalue_counter +=1 #positive eigenvalues
            
            if negative_eigenvalue_counter == 1 and positive_eigenvalue_counter == 2:
                mirror_counter += 1 #pure mirror counter
            
            index +=1
                
        if pos_det_counter == 4: #if only pure rotations
            if order2_counter == 3: #if three two fold rotation axes
                return '222'
            if order4_counter == 2: #if one four fold rotation axis
                return '4'
            
        if order4_counter == 2: #if one four fold roto inversion axis
            return '-4'
        
        if mirror_counter == 2: #if two mirror planes 
            return 'mm2'
        
        if mirror_counter == 1: #if one mirror planes
            return '2/m'
    
    if pg_order == 8:
        pos_det_counter = 0 #tracks number of pure rotations
        order2_counter = 0 #tracks number of order 2 elements
        order4_counter = 0 #tracks number of order 4 elements
        mirror_counter = 0 #tracks number of pure mirrors
        index = 0
        
        for pge in pg:
            
            negative_eigenvalue_counter = 0 #tracks number of negative eigenvalues
            positive_eigenvalue_counter = 0 #tracks number of positive eigenvalues
            
            if pge.det() == 1:
                pos_det_counter +=1 #increments by one for each pure rotation
                
            if pg_orders[index] == 2:
                order2_counter += 1 #increments by one for each order 2 element
            if pg_orders[index] == 4:
                order4_counter += 1 #increments by one for each order 4 element
                
            for eigen_tuple in eigen[index]:
                if eigen_tuple[0] == -1:
                    negative_eigenvalue_counter +=1 #negative eigenvalues
                if eigen_tuple[0] == 1:
                    positive_eigenvalue_counter +=1 #positive eigenvalues
            
            if negative_eigenvalue_counter == 1 and positive_eigenvalue_counter == 2:
                mirror_counter += 1 #pure mirror counter
            index +=1
        
        if pos_det_counter == 8: #only pure rotations
            return '422'
        
        if mirror_counter == 3: # 3 pure mirrors
            return 'mmm'
        
        if mirror_counter == 4: #4 pure mirrors
            return '4mm'
        
        if mirror_counter == 1: #1 pure mirror
            return '4/m'
            
        if mirror_counter == 2: #2 pure mirrors
            return '-42m'
    
    if pg_order == 6:
        pos_det_counter = 0 #tracks number of pure rotations
        order6_counter = 0 #tracks number of order 6 elements
        order3_counter = 0 #tracks number of order 3 elements
        order2_counter = 0 #tracks number of order 2 elements
        mirror_counter = 0 #tracks number of pure mirrors
        inv_counter = 0 #tracks number of inversions
        index = 0
        
        for pge in pg:
            
            negative_eigenvalue_counter = 0 #tracks number of negative eigenvalues
            positive_eigenvalue_counter = 0 #tracks number of positive eigenvalues
            
            if pge.det() == 1:
                pos_det_counter +=1 #increments by one for each pure rotation
            
            if pg_orders[index] == 6:
                order6_counter += 1 #increments by one for each order 6 element
            
            if pg_orders[index] == 3:
                order3_counter += 1 #increments by one for each order 3 element
            
            if pg_orders[index] == 2:
                order2_counter += 1 #increments by one for each order 2 element
            
            for eigen_tuple in eigen[index]:
                if eigen_tuple[0] == -1:
                    negative_eigenvalue_counter +=1 #negative eigenvalues
                if eigen_tuple[0] == 1:
                    positive_eigenvalue_counter +=1 #positive eigenvalues
            
            if negative_eigenvalue_counter == 1 and positive_eigenvalue_counter == 2:
                mirror_counter += 1 #pure mirror counter
                
            if negative_eigenvalue_counter == 3:
                inv_counter += 1 #inversion tracker
                
            index +=1
        
        if pos_det_counter == 6:
            if order6_counter == 2: #6 fold rotation 
                return '6'
            
            if order3_counter == 2 and order2_counter == 3: #3 fold and 2 fold rotation
                return '32'
        
        if mirror_counter == 0: #3 fold rotoinversion
            return '-3'
        
        if mirror_counter == 1: #6 fold rotoinversion
            return '-6'
        
        if mirror_counter == 3: #mirror plane perpendicular to 3 fold rotation axis
            return '3m'
        
    if pg_order == 12:
        pos_det_counter = 0 #tracks number of pure rotations
        order6_counter = 0 #tracks number of order 6 elements
        order3_counter = 0 #tracks number of order 3 elements
        order2_counter = 0 #tracks number of order 2 elements
        mirror_counter = 0 #tracks number of pure mirrors
        inv_counter = 0 #tracks number of inversions
        index = 0
        
        for pge in pg:
            
            negative_eigenvalue_counter = 0 #tracks number of negative eigenvalues
            positive_eigenvalue_counter = 0 #tracks number of positive eigenvalues
            
            if pge.det() == 1:
                pos_det_counter +=1 #increments by one for each pure rotation
            
            if pg_orders[index] == 6:
                order6_counter += 1 #increments by one for each order 6 element
            
            if pg_orders[index] == 3:
                order3_counter += 1 #increments by one for each order 3 element
            
            if pg_orders[index] == 2:
                order2_counter += 1 #increments by one for each order 2 element
            
            for eigen_tuple in eigen[index]:
                if eigen_tuple[0] == -1:
                    negative_eigenvalue_counter +=1 #negative eigenvalues
                if eigen_tuple[0] == 1:
                    positive_eigenvalue_counter +=1 #positive eigenvalues
            
            if negative_eigenvalue_counter == 1 and positive_eigenvalue_counter == 2:
                mirror_counter += 1 #pure mirror counter
                
            if negative_eigenvalue_counter == 3:
                inv_counter += 1 #inversion tracker
                
            index +=1
            
        if pos_det_counter == 12:
            if order2_counter == 3: #23 contains 3 order 2 elements
                return '23'
        
            if order2_counter == 7: #622 contains 7 order 2 elements
                return '622'
            
        if inv_counter == 1:
            
            if mirror_counter == 1: #1 pure mirror
                return '6/m'
            
            if mirror_counter == 3: #3 pure mirrors
                return '-3m' 
        
        if mirror_counter == 6: #6 pure mirrors
            return '6mm'
        
        if mirror_counter == 4: #4 pure mirrors
            return '-6m2' 
            
            
    if pg_order == 24:
        pos_det_counter = 0 #tracks number of pure rotations
        order6_counter = 0 #tracks number of order 6 elements
        order3_counter = 0 #tracks number of order 3 elements
        order2_counter = 0 #tracks number of order 2 elements
        mirror_counter = 0 #tracks number of pure mirrors
        inv_counter = 0 #tracks number of inversions
        index = 0
        
        for pge in pg:
            
            negative_eigenvalue_counter = 0 #tracks number of negative eigenvalues
            positive_eigenvalue_counter = 0 #tracks number of positive eigenvalues
            
            if pge.det() == 1:
                pos_det_counter +=1 #increments by one for each pure rotation
            
            if pg_orders[index] == 6:
                order6_counter += 1 #increments by one for each order 6 element
            
            if pg_orders[index] == 3:
                order3_counter += 1 #increments by one for each order 3 element
            
            if pg_orders[index] == 2:
                order2_counter += 1 #increments by one for each order 2 element
            
            for eigen_tuple in eigen[index]:
                if eigen_tuple[0] == -1:
                    negative_eigenvalue_counter +=1 #negative eigenvalues
                if eigen_tuple[0] == 1:
                    positive_eigenvalue_counter +=1 #positive eigenvalues
            
            if negative_eigenvalue_counter == 1 and positive_eigenvalue_counter == 2:
                mirror_counter += 1 #pure mirror counter
                
            if negative_eigenvalue_counter == 3:
                inv_counter += 1 #inversion tracker
                
            index +=1
    
        if pos_det_counter == 24: #only order 24 group with all pure rotations
            return '432'
        
        if mirror_counter == 7: #7 pure mirrors
            return '6/mmm'
        
        if mirror_counter == 6: #6 pure mirrors
            return '-43m'
        
        if mirror_counter == 3: #3 pure mirrors
            return 'm-3'
            

def group_atoms(atom_point_groups, eigen, atom_name):
    """returns dictionary with atom's point groups grouped based on their point group label
    
    Variables
    
    atom_point_groups - a dictionary with the coordinate_index as value and point group in the form 
    (Rot, timeinv) if time_inv included and of form (Rot,) if otherwise.
    
    eigen - a dicitionary with atom_index as key and real (eigenvalue, eigenvector) tuples as values
    
    atom_name - string label for atom"""
    
    grouped_pgs = {} #stores grouped point groups
    pg_labels = {} #stores labels for each point group
    symbol_list = [] #stores each point group symbol found
    
    for atom_index in atom_point_groups: #loops through all the atom_indices
        pg = [pge[0] for pge in atom_point_groups[atom_index]] #extracts rotation matrices
        symbol = find_pg_symbol(pg, eigen[atom_index], find_pg_orders(pg)) #finds point group symbol
        pg_labels[atom_index] = symbol #stores symbol of pg at appropriate atom index
        symbol_list.append(symbol) #adds symbol to list of all symbols
    
    
    for symbol in symbol_list: #loops through all symbols found
        grouped_pgs[f'{atom_name}_{symbol}'] = {} #creates dictionary for each symbol
        
    for atom_index in pg_labels: #loops through all the atom_indices
        for atom_type in grouped_pgs: #loops through each atom_type
            if pg_labels[atom_index] in atom_type: #stores point group in appropriate atom_type value
                grouped_pgs[f'{atom_name}_{pg_labels[atom_index]}'][atom_index] = atom_point_groups[atom_index]
    return grouped_pgs
        

def group_pgs(elements_of_interest, point_groups, eigen):
    """Take a dictionary of point groups for each atom and returns a dictionary, grouped_pgs where the 
    atoms are grouped based on their point group symbol, and returns a list of tuples, mappings, showing 
    what new atom label each old atom label is mapped to
    
    Variables
    
    elements_of_interest - list of string labels of desired elements
    
    point_groups - a nested dictionary with atom labels as strings as keys and the coordinate_index as 
    sub keys and as values the point group in the form (Rot, timeinv) if time_inv included and of form 
    (Rot,) if otherwise.
    
    eigen - a nested dicitionary with atom_index as key and real (eigenvalue, eigenvector) tuples as values"""
    
    grouped_pgs = {}
    mappings = []
    
    for element in elements_of_interest: #loops through desired elements
        element_grouped_pgs = group_atoms(point_groups[element], eigen[element], element) #groups atoms by 
        #pg symbol
        
        for atom_type in element_grouped_pgs: #loops through each pg symbol
            grouped_pgs[atom_type] = element_grouped_pgs[atom_type] #adds to main dictionary
            mappings.append((element, atom_type))
    
    return grouped_pgs, mappings


def group_coords(mappings, old_coords, grouped_pgs):
    """returns a nested dictionary with the new atom labels as keys and the atom site index as sub key
    with the 3x1 sympy matrix as the value representing the fractional coordinates. 
    
    Variables
    
    mappings - a list of tuples, mappings, showing what new atom label each old atom label is mapped to
    
    old_coordinates - coordinates - dictionary with keys being elements and values being nested dictionaries 
    with integers as keys and lists of coordinates as values of the form [x,y,z] and returns dictionary with 
    elements as keys and coordinates as values and a list of the element labels as strings
    
    grouped_pgs - a dictionary, grouped_pgs where the atoms are grouped based on their point group symbol
    """
    
    new_coords = {}
    
    for mapping in mappings: #loops through old label, new label pairs
        new_coords[mapping[1]] = {} #creates nested dictionary for each new label
        
        for grouped_pg_index in grouped_pgs[mapping[1]]: #loops through coordinate indices in grouped_pgs
            new_coords[mapping[1]][grouped_pg_index] = old_coords[mapping[0]][grouped_pg_index]
    
    return new_coords


def wv_pge_finder(pg, wv_choices_list):
    """Returns list of tuples for w and v if given where the tuples contained the point group element from
    which they were taken and the order of the point group element in the form (pge, order)
    
    Takes the point group of the site used to calculate the first set of axes and wv_choices list which is
    a list of tuples of the form (pge index, order)
    
    Note that if there were no choices for the high and low symmetry axis this function will return the 
    identity matrix and its order 1 for both w and v by default.
    """
    
    if not wv_choices_list: #returns identity if there were no point group elements used
        return [(sympy.eye(3), 1), (sympy.eye(3), 1)]
    
    wv_pges = []
    
    for choice in wv_choices_list:
        wv_pges.append((pg[choice[0]], choice[1]))
    
    return wv_pges


def xyz_wv_pges(wv_pges):
    """Returns a table of the point group elements from which w and v are chosen in the x,y,z form along 
    with the atom label and order of the element and the point group symbol. Takes as input an dictionary 
    wv_pges with atom label strings as keys and lists of tuples of (point group element, order) as values 
    where thefirst entry corresponds to the w axis and the second axis if given corresponds to the v axis. 
    Both entries will be the identity matrix with its order 1 if no suitable symmetry axis could be found. 
    Atom string label should not have an underscore in it other than the one put into it by the group atoms
    function."""
    
    xyz_wv_table = pd.DataFrame(columns=['atom_type', 'point group symbol', 'w axis(x,y,z)', 'w axis order',
                                        'v axis(x,y,z)', 'v axis order'])
    
    mapping = {0:'x', 1:'y', 2:'z'}
    
    for atom_type in wv_pges: #loops through each atom label
        wv_xyz_forms = []
        
        for pge, order in wv_pges[atom_type]: #loops through (pg element, order) tuples
            xyz_form = ''
            rows = [pge[:3], pge[3:6], pge[6:9]] #separates pge into rows
            
            for row in rows: #loops through rows
                index = 0
                
                for element in row: #maps each row element to x,y or z based on index
                    if element == 1:
                         xyz_form += '+' + mapping[index]
                    if element == -1:
                         xyz_form += '-' + mapping[index]
                    index +=1
                    
                xyz_form += ',' #separates each row by comma
                
            xyz_form = xyz_form[:-1] #removes extra comma
            
            wv_xyz_forms.append((xyz_form, order)) #adds x,y,z to list
        
        reverse_atom_type_list = list(atom_type)
        
        reverse_atom_type_list.reverse() #reverses list of atom_type string characters so that first 
        #occurring _ is the one before the appended pg symbol
        
        _index = len(atom_type) - reverse_atom_type_list.index('_') #obtains starting index of point group 
        #symbol
        
        pg_symbol = atom_type[_index:] #finds pg symbol as it should be after 
        #underscore in atom_type string
        
        if len(wv_xyz_forms) == 1: #if no pg element from group was used for v
            xyz_wv_table.loc[len(xyz_wv_table)] = [atom_type, pg_symbol, wv_xyz_forms[0][0], wv_xyz_forms[0][1],
                                                   '-', '-']
            
        if len(wv_xyz_forms) == 2: #if a pg element was used for v
            xyz_wv_table.loc[len(xyz_wv_table)] = [atom_type, pg_symbol, wv_xyz_forms[0][0], wv_xyz_forms[0][1],
                                                   wv_xyz_forms[1][0], wv_xyz_forms[1][1]]
    return xyz_wv_table

def eigen_frac2cart(eigen_dict, lattice_vector_table):
    """takes in a nested dictionary with elements as keys and point groups as sub keys and point group element indices as sub
    sub keys with lists of (eigenvalue, eigenvector) tuples as values with eigenvectors in fractional coordinates and returns
    a similar nested dictionary where the eigenvectors are in cartesian coordinates
    
    also takes as input a pandas DataFrame with columns of Lattice_vector, x, y, z and Lattice_parameter which are used to 
    convert the fractional coordinates to cartesian"""
    
    cart_eigen_dict = {}
    
    transform = get_transform_from_lattice_vectors(lattice_vector_table) #transforms fractional coordinates to cartesian
    
    for element in eigen_dict: #loops through elements
        cart_eigen_dict[element] = {}
        
        for pg_index in eigen_dict[element]: #loops through each site
            cart_eigen_dict[element][pg_index] = {}
            
            for pge_index in eigen_dict[element][pg_index]:#loops through each point group element
                cart_eigen_dict[element][pg_index][pge_index] = []
                
                for eigenvalue, eigenvector in eigen_dict[element][pg_index][pge_index]: #loops through each (eigenvalue,eigenvector) tuple
                    cart_eigen_dict[element][pg_index][pge_index].append((eigenvalue, transform * eigenvector))
    
    return cart_eigen_dict
    
def prop_tensor_dict(elements_of_interest, pgs_dict, highest_rank):
    """returns a nested dictionary with the elements of interest as keys and the site indices as sub keys with 
    a list of the even magnetocrystalline anisotropy tensors generated from the point groups of each site up to 
    the highest rank as values.
    
    Variables
    
    elements of interest - list of string labels of desired elements
    
    pgs_dict - nested dictionary with elements as keys and site indices as sub keys and lists of the point group
    elements as values where the point group elements are tuples either of the form (point group op,) if no time
    inversion is included or of the form (point group op, time inversion) if time inversion is included. point 
    group op must be a sympy 3x3 Matrix.
    
    highest_rank - highest rank of property tensors to be calculated. must be int
    
    """
    
    if type(highest_rank) != int or highest_rank <= 0 or highest_rank%2 != 0:
        raise ValueError('highest_rank must be a positive even integer')
    
    property_tensors_dict = {}
    
    for element in elements_of_interest: #loops through elements of interest
        property_tensors_dict[element] = {}
        
        for pg_index in pgs_dict[element]: #loops through each site for element
            property_tensors_dict[element][pg_index] = []
            
            pges = [pge[0] for pge in pgs_dict[element][pg_index]] #collects point group elements
            
            for rank in range(2, highest_rank+1, 2): #loops through all possible ranks
            
                #adds property tensor for each rank to list
                property_tensors_dict[element][pg_index].append(crystal_tensor(rank, *pges)) 
            
            print(f"site {pg_index+1}'s property tensors done out of {len(pgs_dict[element])} for {element}")
            
    return property_tensors_dict

def prop_table(tensor):
    """returns a pandas DataFrame with the a column for each index of the tensor (number of columns depends on dimensions of tensor) and 
    a column with a string entry of the value of the tensor at that location. takes an even rank tensor represented as a sympy array as input.
    Only the non zero values are stored in the table.
    """
    
    rank = len(tensor.shape)
    
    prop_table = pd.DataFrame(columns=[f'index{index}' for index in range(1, rank+1)]+['value'])
    
    for array_indices in product(*[range(dim) for dim in tensor.shape]): #loops through each possible tuple of indices
        indices = [index+1 for index in array_indices]
        
        if tensor[array_indices] != 0: #only stores non zero values
            prop_table.loc[len(prop_table)] = [*indices, str(tensor[array_indices])]
    
    return prop_table

def write_prop_tensors_file(tensor_dict, coord_dict, filename):
    """writes a csv file where a table is written for each tensor of each site. Takes nested dictionary tensor_dict with the elements of 
    interest as keys and the site indices as sub keys with a list of the tensors for each site, a nested dictionary coord_dict where the elements are keys, the site index is the sub key and 
    the values are sympy 3x1 Matrices representing the fractional coordinates of the site and a filename which is a string which is 
    the name of file to which the csv will be written"""
    
    tensor_table_file = open(f'{filename}.csv', 'w')
    
    for element in tensor_dict: #loops through each element
        tensor_table_file.write(f'{element}\n\n') #writes element label
        
        for site_index in tensor_dict[element]: #loops through each site
            coord = coord_dict[element][site_index]
            tensor_table_file.write('Fractional Coordinates\n')
            tensor_table_file.write('x,y,z\n')
            tensor_table_file.write(f'{coord[0]},{coord[1]},{coord[2]}\n\n') #writes site's fractional coordinates
            
            for tensor in tensor_dict[element][site_index]: #loops through each tensor for each site
                rank = len(tensor.shape)
                tensor_table_file.write(f'Rank {rank} entries\n')
                tensor_table = prop_table(tensor) #creates table for prop tensor
                for col in list(tensor_table.columns):
                    if col != 'value':
                        tensor_table_file.write(col+',')
                    else:
                        tensor_table_file.write(col+'\n')
                
                for index, row in tensor_table.iterrows():
                    
                    for entry in row: #writes out entries of table
                        if type(entry) == int:
                            tensor_table_file.write(str(entry)+',') 
                        else:
                            tensor_table_file.write(entry + '\n\n')
                            
            tensor_table_file.write('-,-,-,-,-\n\n')
        
        tensor_table_file.write('-,-,-,-,-,-,-,-,-,-\n\n')
            
def get_atol(cif_dict, key = "_atom_site_label"):
    """returns absolute tolerance value as a float based on the precision of the fractional coordinates given in the cif file. cif_file is a dicitonary
    like object read into the programme using ReadCif and the absolute tolerance is returned as the same order of magnitude as the precision of the
    most precise fractional coordinate given. Key specifies where in the cif file to find the fractional coordinates of each element. Set to 
    "_atom_site_label" as default."""
    
    len_list = []
    
    for index, element in enumerate(cif_dict[key]): #loops through elements in cif file
        
        for coord in ['x','y','z']: #loops through each coord
            len_list.append(len(cif_dict[f'_atom_site_fract_{coord}'][index]))
    
    max_pres = max(len_list) #stores highest precision by number of decimal places + length of string '0.' if included
    
    if max_pres <= 3:
        return 10**(-8) #if highest precision coordinate was 0 or 0.0 returns np.isclose default precision
    
    else:
        return 10**(-(max_pres-2)) #returns an order of magnitude of highest precision (-2 accounts for '0.' in string)
       
def make_coord_table(all_coords_dict):
    """returns a pandas DataFrame with the columns atom_label, x, y and z where atom_label is the string label for the
    atom as given in the CIF file and x,y and z are the fractional coordinates of the atoms. Takes a nested dictionary 
    all_coord_dict as input where the elements are keys, the site index is the sub key and the values are sympy 3x1 
    Matrices representing the fractional coordinates of the site. """
    
    coord_table = pd.DataFrame(columns = ['atom_label', 'x', 'y', 'z'])
    
    for atom_label in all_coords_dict: #loops through each atom_label type
        for coord_index in all_coords_dict[atom_label]: #loops through all the atoms site indices
            coord = all_coords_dict[atom_label][coord_index] #stores 3x1 sympy matrix fractional coordinate vector
            coord_table.loc[len(coord_table)] = [atom_label, coord[0], coord[1], coord[2]]
            
    return coord_table

def get_transform_from_lattice_vectors(lattice_vector_table):
    """Returns 3x3 sympy Matrix which transforms fractional coordinates to Cartesian. Takes a Pandas DataFrame object 
    lattice_vectors which contains the lattice vectors and lattice parameters of the cell."""
    
    transform = np.zeros((3,3), dtype=float) 
    
    for i in range(3): #creates transformation matrix from lattice vectors
        transform[:,i] = lattice_vector_table.iloc[i][4] * np.array(lattice_vector_table.iloc[i][1:4], dtype=float)
    
    return sympy.Matrix(transform)

#obtained from https://physics.stackexchange.com/questions/351372/generate-all-elements-of-a-point-group-from-generating-set

def generate_pg(N,*args):
    """returns list of 3x3 rotation matrices for a point group generated from *args, which are generators
    for the point group and N is the order of the pg"""
    
    # e: identity operator
    # G: list of generators
    # L: list of all the elements of the group
    # N: maximum order of the group we deem acceptable
    G = []
    for arg in args:
        G.append(arg)
        
    e = sympy.eye(3)
    g = g1 = G[0]
    L = [e]
    
    while g != e:
        L.append(g)
        assert len(L) <= N
        g = g*g1
    for i in range(1,len(G)):
        C = [e]
        L1 = list(L)
        more = True
        while more:
            assert len(L) <= N
            more = False
            for g in list(C):
                for s in G[:i+1]:
                    sg = s*g
                    if sg not in L:
                        C.append(sg)
                        L.extend([ sg*t for t in L1 ])
                        more = True
    return L

#generator matrices
M7 = sympy.Matrix([[0,-1,0],[1,0,0],[0,0,1]])
M14 = sympy.Matrix([[0,-1,0],[0,0,-1],[-1,0,0]])
M11 = sympy.Matrix([[Fraction(1,2),-sympy.sqrt(3)/2,0],
                    [sympy.sqrt(3)/2,Fraction(1/2),0],
                    [0,0,1]])
M5 = sympy.Matrix([[-1,0,0],[0,1,0],[0,0,1]])
M3 = sympy.Matrix([[1,0,0],[0,1,0],[0,0,-1]])
M8 = sympy.Matrix([[0,-1,0],
                    [1,0,0],
                    [0,0,-1]])
M10 = sympy.Matrix([[Fraction(1/2), -sympy.sqrt(3)/2,0],
                    [sympy.sqrt(3)/2, Fraction(1/2),0],
                    [0,0,-1]])
M7 = sympy.Matrix([[0,-1,0],
                  [1,0,0],
                  [0,0,1]])
M6 = sympy.Matrix([[1,0,0],
                  [0,-1,0],
                  [0,0,1]])
M0 = sympy.eye(3)

M1 = -sympy.eye(3)

M2 = sympy.Matrix([[-1,0,0],
                  [0,-1,0],
                  [0,0,1]])

M4 = sympy.Matrix([[1,0,0],
                  [0,-1,0],
                  [0,0,-1]])

M9 = sympy.Matrix([[-Fraction(1,2),-sympy.sqrt(3)/2,0],
                    [sympy.sqrt(3)/2,-Fraction(1/2),0],
                    [0,0,1]])

M12 = sympy.Matrix([[-Fraction(1,2),-sympy.sqrt(3)/2,0],
                    [sympy.sqrt(3)/2,-Fraction(1/2),0],
                    [0,0,-1]])

M13 = sympy.Matrix([[0,0,1],
                  [1,0,0],
                  [0,1,0]])