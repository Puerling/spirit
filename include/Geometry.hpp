#pragma once

#include <Vectormath_Defines.hpp>

// Geometry contains all geometric information of a system
struct Geometry
{
    // ---------- Constructor
    //  Build a regular lattice from a defined basis cell and translations
    Geometry( intfield n_cells_, int n_cell_atoms_ )
        : n_cells( std::move( n_cells_ ) ),
          n_cell_atoms( n_cell_atoms_ ),
          nos( n_cell_atoms * n_cells[0] * n_cells[1] * n_cells[2] ),
          nos_nonvacant( n_cell_atoms * n_cells[0] * n_cells[1] * n_cells[2] ),
          n_cells_total( n_cells[0] * n_cells[1] * n_cells[2] ),
          atom_types( this->nos, 0 )
    {
    };

    // Number of cells {na, nb, nc}
    intfield n_cells;
    // Number of spins per basic domain
    int n_cell_atoms;

    // ---------- Inferrable information
    // Number of sites (total)
    int nos;
    // Number of non-vacancy sites (if defects are activated)
    int nos_nonvacant;
    // Number of basis cells total
    int n_cells_total;
    // Atom types of all the atoms: type index 0..n or or vacancy (type < 0)
    intfield atom_types;
};
