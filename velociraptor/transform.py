"""Tools to compute the linear mapping between environments and dipoles/charges"""


import numpy as np


def transform_envts_charge_dipoles(molecules, target_matrix,
                                   transpose_kernel=False,
                                   dipole_normalize=False):
    """Transform a matrix of environments to charges+dipoles

    Parameters:
        molecules       A list of ASE Atoms objects containing the
                        atomic positions for each configuration
        target_matrix   The matrix to transform; rows should correspond
                        to environments (atoms).  The column dimension
                        is retained in the output (left-multiplication).
    Optional parameters:
        transpose_kernel
                        Whether the kernel was computed in the opposite
                        order, sparse:full (MN) instead of the expected
                        full:sparse (NM) -- in this case, meaning the
                        environments are along the columns rather than
                        the rows.
        dipole_normalize
                        Whether to normalize the molecular kernel by the
                        number of atoms, in order to work with dipole
                        moments and charges that are similarly normalized

    This performs the product L@K, where the matrix L is composed of
    blocks arranged diagonally, each block corresponding to a single
    molecule and having the form:

       [1   1   1   ... 1  ]
       [x_1 x_2 x_3 ... x_n]
       [y_1 y_2 y_3 ... y_n]
       [z_1 z_2 z_3 ... z_n]

    (the indices label atoms within the current molecule), and K is the
    target matrix.

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
        target_shape_new = (4 * len(molecules), )
    elif len(target_matrix.shape) > 1:
        if transpose_kernel:
            target_matrix = target_matrix.transpose()
        target_shape_new = (4 * len(molecules), target_matrix.shape[1])
    else:
        raise ValueError("Cannot operate on a 0-d array")
    if target_matrix.shape[0] != sum(len(mol)
                                     for mol in molecules):
        raise ValueError("Target matrix must have as many rows (columns, if "
                         "transposed) as environments (i.e. atoms) in the "
                         "list of molecules")
    target_transformed = np.empty(target_shape_new)
    environ_idx = 0
    for mol_idx, molecule in enumerate(molecules):
        natoms_mol = len(molecule)
        molecule_target = target_matrix[environ_idx:environ_idx+natoms_mol]
        environ_idx += natoms_mol
        molecule_positions = molecule.get_positions()
        molecule_positions -= np.mean(molecule_positions, axis=0)
        molecule_trafo = np.concatenate((np.ones((natoms_mol, 1)),
                                         molecule_positions),
                                        axis=1)
        molecule_target_transformed = molecule_trafo.T.dot(molecule_target)
        if dipole_normalize:
            molecule_target_transformed /= natoms_mol
        target_transformed[
            mol_idx*4:(mol_idx+1)*4] = molecule_target_transformed
    return target_transformed


def transform_envts_partial_charges(molecules, target_matrix,
        transpose_kernel=False):
    """Transform a matrix of environments to atomic partial charges

    Parameters:
        molecules       A list of ASE Atoms objects containing the
                        atomic positions for each configuration
        target_matrix   The matrix to transform; rows should correspond
                        to environments (atoms).  The column dimension
                        is retained in the output (left-multiplication).
    Optional parameters:
        transpose_kernel
                        Whether the kernel was computed in the opposite
                        order, sparse:full (MN) instead of the expected
                        full:sparse (NM) -- in this case, meaning the
                        environments are along the columns rather than
                        the rows.
    """
    if transpose_kernel:
        target_transformed = target_matrix.transpose()
    else:
        target_transformed = target_matrix
    if target_transformed.shape[0] != sum(len(mol)
                                          for mol in molecules):
        raise ValueError("Target matrix must have as many rows (columns, if "
                         "transposed) as environments (i.e. atoms) in the "
                         "list of molecules")
    # Welp, that was easy (the scalar kernel matrix is already in terms of
    # partial charges)
    return target_transformed


def transform_spherical_tensor_to_cartesian(target_matrix):
    """Transform a matrix from spherical-tensor to Cartesian ordering"""
    target_shape = target_matrix.shape
    if target_shape[2:] != (3, 3):
        raise ValueError('Vector kernel has unrecognized shape: {}'.format(
                target_shape))
    return target_matrix[:, :, [2, 0, 1]][:, :, :, [2, 0, 1]]


def transform_vector_envts_charge_dipoles(
        molecules, target_matrix, transpose_kernel=False,
        dipole_normalize=True, spherical=False):
    """Transform a matrix of vector environments to charges+dipoles

    This takes a matrix of shape (n_envs, n_sparse, 3, 3), describing
    the vector components of the kernels between individual atoms, to
    another matrix of shape (n_molecules*4, n_sparse*3) describing the
    kernel of the vector components of the _molecular_ dipole with the
    components of the sparse environments.  Every fourth row (starting
    with the first row) is zero, since the correlation between an l=1
    tensor and a scalar (l=0) is zero: Atomic dipoles predict no charges.

    Parameters:
        molecules       A list of ASE Atoms objects containing the
                        atomic positions for each configuration
        target_matrix   The matrix to transform; rows should correspond
                        to environments (atoms).  The column dimension
                        is retained in the output (left-multiplication).
    Optional parameters:
        transpose_kernel
                        Whether the kernel was computed in the opposite
                        order, sparse:full (MN) instead of the expected
                        full:sparse (NM)
        dipole_normalize
                        Whether to normalize the molecular kernel by the
                        number of atoms, in order to work with dipole
                        moments and charges that are similarly normalized
        spherical       Whether the kernel matrix was stored in "spherical
                        tensor" order (i.e. in order of the spherical l=1
                        components, rather than the Cartesian components)
                        Setting this option to True will transform it to
                        the standard Cartesian basis used by the rest of
                        velociraptor

    The charges and dipoles are ordered canonically, i.e. for each
    configuration the total charge first and then the three Cartesian
    dipole components.
    """
    target_shape = target_matrix.shape
    if target_shape[2:] != (3, 3):
        raise ValueError('Vector kernel has unrecognized shape: {}'.format(
                target_shape))
    if not transpose_kernel:
        target_n_envs = target_matrix.shape[0]
        target_shape_new = (len(molecules)*4, target_shape[1] * 3)
    else:
        target_n_envs = target_matrix.shape[1]
        target_shape_new = (len(molecules)*4, target_shape[0] * 3)
    if target_n_envs != sum(len(mol)
                            for mol in molecules):
        raise ValueError("Target matrix must have as many rows (columns, if "
                         "transposed) as environments (i.e. atoms) in the "
                         "list of molecules")
    if spherical:
        target_matrix = target_matrix[:, :, [2, 0, 1]][:, :, :, [2, 0, 1]]
    target_transformed = np.empty(target_shape_new)
    environ_idx = 0
    for mol_idx, molecule in enumerate(molecules):
        natoms_mol = len(molecule)
        # Charge correlation iz sero
        target_transformed[4*mol_idx] = 0.
        if not transpose_kernel:
            molecule_target = target_matrix[environ_idx:environ_idx+natoms_mol]
            target_transformed[4*mol_idx + 1 : 4*mol_idx + 4] = (
                molecule_target.mean(axis=0).transpose(1, 0, 2).reshape(3, -1))
        else:
            molecule_target = target_matrix[
                    :, environ_idx:environ_idx+natoms_mol]
            if dipole_normalize:
                molecule_target_summed = molecule_target.mean(axis=1)
            else:
                molecule_target_summed = molecule_target.sum(axis=1)
            target_transformed[4*mol_idx + 1 : 4*mol_idx + 4] = (
                molecule_target_summed.transpose(2, 0, 1).reshape(3, -1))
        environ_idx += natoms_mol
    return target_transformed


def transform_vector_envts_atomic_dipoles(
        molecules, target_matrix, transpose_kernel=False, spherical=False):
    """Transform a matrix of vector molecular kernels to atomic dipoles

    This takes a matrix of shape (n_envs, n_sparse, 3, 3), describing
    the vector components of the kernels between individual atoms, to
    another matrix of shape (n_envs*3, n_sparse*3) describing the
    kernel of the Cartesian vector components of the atomic dipoles with
    the components of the sparse environments.

    Parameters:
        molecules       A list of ASE Atoms objects containing the
                        atomic positions for each configuration
        target_matrix   The matrix to transform; rows should correspond
                        to environments (atoms).  The column dimension
                        is retained in the output (left-multiplication).
    Optional parameters:
        transpose_kernel
                        Whether the kernel was computed in the opposite
                        order, sparse:full (MN) instead of the expected
                        full:sparse (NM)
        spherical       Whether the kernel matrix was stored in "spherical
                        tensor" order (i.e. in order of the spherical l=1
                        components, rather than the Cartesian components)
                        Setting this option to True will transform it to
                        the standard Cartesian basis used by the rest of
                        velociraptor
    """
    target_shape = target_matrix.shape
    if target_shape[2:] != (3, 3):
        raise ValueError('Vector kernel has unrecognized shape: {}'.format(
                target_shape))
    if not transpose_kernel:
        target_n_envs = target_matrix.shape[0]
        target_shape_new = (target_n_envs * 3, target_shape[1] * 3)
    else:
        target_n_envs = target_matrix.shape[1]
        target_shape_new = (target_n_envs * 3, target_shape[0] * 3)
    if target_n_envs != sum(len(mol)
                            for mol in molecules):
        raise ValueError("Target matrix must have as many rows (columns, if "
                         "transposed) as environments (i.e. atoms) in the "
                         "list of molecules")
    if spherical:
        target_matrix = target_matrix[:, :, [2, 0, 1]][:, :, :, [2, 0, 1]]
    target_new = target_matrix.transpose((0, 2, 1, 3))
    if not transpose_kernel:
        target_new = target_new.reshape(target_shape_new)
    else:
        target_new = target_new.reshape(target_shape_new[::-1]).transpose()
    return target_new


def transform_vector_mols_charge_dipoles(
        molecules, target_matrix, transpose_kernel=False,
        dipole_unnormalize=False, spherical=False):
    """Transform a matrix of vector molecular kernels to charges+dipoles

    This takes a matrix of shape (n_mol, n_sparse, 3, 3), describing the
    vector components of the kernels from molecules to sparse
    environments, to another matrix of shape (n_molecules*4, n_sparse*3)
    describing the kernel of the vector components of the molecular
    dipole with the components of the sparse environments.  Every fourth
    row (starting with the first row) is zero, since the correlation
    between an l=1 tensor and a scalar (l=0) is zero: Atomic dipoles
    predict no charges.

    Parameters:
        molecules       A list of ASE Atoms objects containing the
                        atomic positions for each configuration
        target_matrix   The matrix to transform; rows should correspond
                        to molecules.  The column dimension is retained
                        in the output (left-multiplication).
    Optional parameters:
        transpose_kernel
                        Whether the kernel was computed in the opposite
                        order, sparse:full (MN) instead of the expected
                        full:sparse (NM)
        dipole_unnormalize
                        Whether to undo the (default) normalization of the
                        molecular vector kernel by the number of atoms, in
                        order to work with dipole moments that are
                        likewise not normalized
        spherical       Whether the kernel matrix was stored in "spherical
                        tensor" order (i.e. in order of the spherical l=1
                        components, rather than the Cartesian components)
                        Setting this option to True will transform it to
                        the standard Cartesian basis used by the rest of
                        velociraptor

    The charges and dipoles are ordered canonically, i.e. for each
    configuration the total charge first and then the three Cartesian
    dipole components.
    """
    target_shape = target_matrix.shape
    if target_shape[2:] != (3, 3):
        raise ValueError('Vector kernel has unrecognized shape: {}'.format(
                target_shape))
    if not transpose_kernel:
        target_n_molecules = target_matrix.shape[0]
        target_shape_new = (len(molecules)*4, target_shape[1] * 3)
    else:
        target_n_molecules = target_matrix.shape[1]
        target_shape_new = (len(molecules)*4, target_shape[0] * 3)
    if target_n_molecules != len(molecules):
        raise ValueError("Target matrix must have as many rows (columns, if "
                         "transposed) as molecules provided")
    if spherical:
        target_matrix = target_matrix[:, :, [2, 0, 1]][:, :, :, [2, 0, 1]]
    if dipole_unnormalize:
        natoms_mol = np.array([len(mol) for mol in molecules])
        if not transpose_kernel:
            target_matrix *= natoms_mol.reshape(-1, 1, 1, 1)
        else:
            target_matrix *= natoms_mol.reshape(1, -1, 1, 1)
    if not transpose_kernel:
        target_new = np.concatenate((np.zeros(target_shape[:2] + (1, 3)),
                                     target_matrix), axis=2)
    else:
        target_new = np.concatenate((np.zeros(target_shape[:2] + (3, 1)),
                                     target_matrix), axis=3)
    target_new = target_new.transpose((0, 2, 1, 3))
    if not transpose_kernel:
        target_new = target_new.reshape(target_shape_new)
    else:
        target_new = target_new.reshape(target_shape_new[::-1]).transpose()
    return target_new


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
    n_environments = sum(len(mol) for mol in molecules)
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
        natoms_mol = len(molecule)
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
