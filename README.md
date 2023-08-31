# hams
Python code for analysing crystal symmetry allowed magnetic anisotropies of individual magnetic sites on a crystal.

# Description
This code is for symbolically calculating the magnetocrystalline anisotropies of individual magnetic sites on a crystal along with generating the optimal choices of uvw axes to be used in the anisotropy energy expression based on the point group of each site. 

## Features
hams has 3 ways of outputting the magnetocrystalline anisotropy energy of each site for an element/s of interest in a magnetic crystal. From a cif or mcif file the code can generate a table giving:

- the components of each magnetocrystalline anisotropy property tensor as sympy variables showing how each component is related, up to a specified rank for each site of the element/s of interest or,
- the magnetocrystalline anisotropy Hamiltonian for the first (closest to origin) magnetic site for a certain element/s of interest in the crystal with the Hamiltonian written in sympy format or,
- the magnetocrystalline anisotropy Hamiltonian for each of the magnetic sites for a certain element/s of interest in the crystal with each Hamiltonian written in sympy format.

Note that each of the above tables can be stored as a csv file and the Hamiltonians can be output in terms of direction cosines either in Cartesian or Spherical Polar coordinates. The code generates 4 more tables that can also be stored as csv files:
- a table containing the lattice vectors and lattice parameters of the crystal,
- a table containing the optimal uvw choices for each site of the element/s of interest on the crystal (chosen based on the point group of the site),
- a table showing which elements of the point group were used to generate the uvw axes, giving each elements' form in x,y,z format, their order and the point group they are from,
- a table containing the coordinates of all of the atoms in the conventional unit cell of the crystal.

hams can also:
- create a vesta file (using an exisiting vesta file for the crystal) to show the uvw axes on each site for an element of interest in the crystal,
- generate a 3D interactive plot for a chosen magnetocrystalline anisotropy energy term, showing the energy density surface at each site for an element of interest.

hams allows you to specify the highest rank of magnetocrystalline anisotropy property tensor to be considered in calculations, specify which elements you want to be analysed in the crystal, and specify the tolerance to be used for comparing fractional coordinates with the numpy.isclose function. hams can handle cif and mcif files with and without time inversion in the symmetry operations, with symmetry operations in both the "rotation then translation" and "translation then rotation" formats, i.e. both "x+1/2" and "1/2+x" formats.

## Acknowledgements

This project was supported by a Royal Society summer studentship in 2023.
