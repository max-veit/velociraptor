"""Tools to compute the linear mapping between environments and dipoles/charges"""


import numpy as np


def transform_envts_charge_dipoles(molecules, target_matrix):
    """Transform a matrix of environments to charges+dipoles

    Parameters:
        molecules       A list of ASE Atoms objects containing the
                        atomic positions for each configuration
        target_matrix   The matrix to transform; rows should correspond
                        to environments (atoms).  The column dimension
                        is retained in the output (left-multiplication).

    The charges and dipoles are ordered canonically, i.e. for each
    configuration the total charge first and then the three Cartesian
    dipole components.

    In-memory version: If 'target_matrix' is specified as a slice of a
    large, on-disk array, the entire thing will in general be read into
    memory -- so make sure you have enough memory to hold 'target_matrix'
    before running this function.

    Note that this function automatically uses the positions relative to
    the centre of geometry of the molecule (mean of the positions of all
    the atoms in the molecule).
    """
    if len(target_matrix.shape) == 1:
        target_transformed = np.empty((4*len(molecules),))
    elif len(target_matrix.shape) > 1:
        target_transformed = np.empty((4*len(molecules),
                                       target_matrix.shape[1]))
    else:
        raise ValueError("Cannot operate on a 0-d array")
    if target_matrix.shape[0] != sum(mol.get_number_of_atoms()
                                     for mol in molecules):
        raise ValueError("Target matrix must have as many rows as " +
                         "environments (atoms) in the list of molecules")
    environ_idx = 0
    for mol_idx, molecule in enumerate(molecules):
        natoms_mol = molecule.get_number_of_atoms()
        molecule_target = target_matrix[environ_idx:environ_idx+natoms_mol]
        environ_idx += natoms_mol
        molecule_positions = molecule.get_positions()
        molecule_positions -= np.mean(molecule_positions, axis=0)
        molecule_trafo = np.concatenate((np.ones((natoms_mol, 1)),
                                         molecule_positions),
                                        axis=1)
        molecule_target_transformed = molecule_trafo.T.dot(molecule_target)
        target_transformed[
            mol_idx*4:(mol_idx+1)*4] = molecule_target_transformed
    return target_transformed


def transform_charge_dipoles_envts(molecules, target_matrix):
    """Transform a matrix of charges+dipoles back to environments

    Parameters:
        molecules       A list of ASE Atoms objects containing the
                        atomic positions for each configuration
        target_matrix   The matrix to transform; rows should correspond
                        to charges and dipoles in the canonical order
                        (for each configuration, one total charge and
                        three Cartesian dipole components).  The column
                        dimension is retained in the output
                        (left-multiplication).

    In-memory version: If 'target_matrix' is specified as a slice of a
    large, on-disk array, the entire thing will in general be read into
    memory -- so make sure you have enough memory to hold 'target_matrix'
    before running this function.

    Note that this function automatically uses the positions relative to
    the centre of geometry of the molecule (mean of the positions of all
    the atoms in the molecule).
    """
    n_environments = sum(mol.get_number_of_atoms() for mol in molecules)
    if len(target_matrix.shape) == 1:
        target_transformed = np.empty((n_environments,))
    elif len(target_matrix.shape) > 1:
        target_transformed = np.empty((n_environments, target_matrix.shape[1]))
    else:
        raise ValueError("Cannot operate on a 0-d array")
    if target_matrix.shape[0] != 4*len(molecules):
        raise ValueError("Target matrix must have as many rows as " +
                         "charge+dipole entries (4 * number of molecules)")
    environ_idx = 0
    for mol_idx, molecule in enumerate(molecules):
        natoms_mol = molecule.get_number_of_atoms()
        molecule_target = target_matrix[mol_idx*4:(mol_idx+1)*4]
        molecule_positions = molecule.get_positions()
        molecule_positions -= np.mean(molecule_positions, axis=0)
        molecule_trafo = np.concatenate((np.ones((natoms_mol, 1)),
                                         molecule_positions),
                                        axis=1)
        molecule_target_transformed = molecule_trafo.dot(molecule_target)
        target_transformed[
            environ_idx:environ_idx+natoms_mol] = molecule_target_transformed
        environ_idx += natoms_mol
    return target_transformed


def transform_h5_envts_charges_dipoles(molecules, target_h5_matrix, row_range):
    """Transform a matrix of environments to charges+dipoles

    Parameters:
        molecules       A list of ASE Atoms objects containing the
                        atomic positions for each configuration
        target_matrix   The matrix to transform; rows should correspond
                        to environments (atoms).  The column dimension
                        is retained in the output (left-multiplication).

    Offline version: The matrix is assumed to exist on-disk (HDF5 format,
    e.g., using h5py), so the matrix reference should be passed in along
    with a tuple to specify the range of rows to use.
    """
    if len(target_h5_matrix.shape) == 1:
        target_transformed = np.empty((len(molecules),))
    elif len(target_h5_matrix.shape) > 1:
        target_transformed = np.empty((len(molecules),
                                       target_h5_matrix.shape[1]))
    else:
        raise ValueError("Cannot operate on a 0-d array")
    if (row_range[1] - row_range[0]) != sum(mol.get_number_of_atoms()
                                            for mol in molecules):
        raise ValueError("Must specify as many rows as environments (atoms)"
                         + " in the list of molecules")
    environ_idx = row_range[0]
    for mol_idx, molecule in enumerate(molecules):
        natoms_mol = molecule.get_number_of_atoms()
        molecule_target = target_h5_matrix[environ_idx:environ_idx+natoms_mol]
        environ_idx += natoms_mol
        molecule_positions = molecule.get_positions()
        molecule_positions -= np.mean(molecule_positions, axis=0)
        molecule_trafo = np.concatenate((np.ones((natoms_mol, 1)),
                                         molecule_positions),
                                        axis=1)
        molecule_target_transformed = molecule_trafo.T.dot(molecule_target)
        target_transformed[
            mol_idx*4:(mol_idx+1)*4] = molecule_target_transformed
    return target_transformed

