"""
io.py — HDF5 persistence layer for the vqlp package.

Provides two module-level recursive helpers (_write_dict, _read_dict) that
serialize/deserialize arbitrary nested dicts whose values may be dense arrays,
sparse CSR matrices, ragged list-of-lists, label arrays (object arrays with
None entries), scalars, None, or further nested dicts.

VQRecallerIO and VQFitterIO use these helpers to save and load fitted
instances. They are the single owners of the HDF5 file structure; no other
module in vqlp imports h5py or knows about the file layout.

Typical usage
-------------
Save a fitted VQFitter::

    VQFitterIO(fitter).save("fitter.h5")

Or via the convenience wrapper on VQFitter itself::

    fitter.save("fitter.h5")

Load back::

    fitter = VQFitterIO.load("fitter.h5")
    # or equivalently:
    fitter = VQFitter.load("fitter.h5")

Similarly for VQRecaller::

    VQRecallerIO(recaller).save("recaller.h5")
    recaller = VQRecallerIO.load("recaller.h5")

Type dispatch
-------------
_write_dict inspects each value and tags it with a ``_type`` HDF5 attribute
so that _read_dict can reconstruct the correct Python type without any
external schema:

    dense        np.ndarray            → dataset   + _type='dense'
    sparse       scipy.sparse.csr      → subgroup  + _type='sparse'
    ragged       list of lists         → subgroup  + _type='ragged'
    label_array  object ndarray/Nones  → dataset   + _type='label_array'
    dict         nested dict           → subgroup  + _type='dict'
    none         None                  → subgroup  + _type='none'
    scalar       int/float/str/bool    → attr  (no _type needed; HDF5 preserves type)
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse, csr_matrix

__all__ = ["VQRecallerIO", "VQFitterIO"]

# ---------------------------------------------------------------------------
# Type tag constants
# ---------------------------------------------------------------------------

_T_DENSE  = "dense"
_T_SPARSE = "sparse"
_T_RAGGED = "ragged"
_T_LABEL  = "label_array"
_T_DICT   = "dict"
_T_NONE   = "none"


# ---------------------------------------------------------------------------
# Module-level recursive helpers
# ---------------------------------------------------------------------------

def _write_dict(group, d):
    """
    Recursively write a dict ``d`` into an open h5py Group ``group``.

    Each value is inspected and tagged with a ``_type`` attribute so that
    _read_dict can reconstruct the correct Python type on load.  Keys must
    be strings.

    Supported value types
    ---------------------
    np.ndarray          → HDF5 dataset, _type='dense'
    scipy sparse CSR    → HDF5 subgroup, _type='sparse'
    list of lists       → HDF5 subgroup, _type='ragged'
    object ndarray      → HDF5 dataset,  _type='label_array'
    dict                → HDF5 subgroup, _type='dict'  (recursive)
    None                → HDF5 subgroup, _type='none'
    scalar              → HDF5 attr (int, float, str, bool; no _type needed)

    Parameters
    ----------
    group : h5py.Group
        Open, writable HDF5 group to write into.
    d : dict
        Dict whose values are of the supported types above.
    """
    import h5py

    for key, value in d.items():
        key = str(key)

        # ---- None --------------------------------------------------------
        if value is None:
            sg = group.create_group(key)
            sg.attrs["_type"] = _T_NONE

        # ---- nested dict -------------------------------------------------
        elif isinstance(value, dict):
            sg = group.create_group(key)
            sg.attrs["_type"] = _T_DICT
            _write_dict(sg, value)

        # ---- sparse CSR --------------------------------------------------
        elif issparse(value):
            sg = group.create_group(key)
            sg.attrs["_type"] = _T_SPARSE
            csr = value.tocsr()
            sg.create_dataset("data",    data=csr.data)
            sg.create_dataset("indices", data=csr.indices)
            sg.create_dataset("indptr",  data=csr.indptr)
            sg.attrs["shape"] = list(csr.shape)
            sg.attrs["dtype"] = str(csr.dtype)

        # ---- object ndarray (label array: entries are scalars or None) ---
        elif isinstance(value, np.ndarray) and value.dtype == object:
            _write_label_array(group, key, value)

        # ---- dense ndarray -----------------------------------------------
        elif isinstance(value, np.ndarray):
            ds = group.create_dataset(key, data=value)
            ds.attrs["_type"] = _T_DENSE

        # ---- ragged list-of-lists ----------------------------------------
        elif (
            isinstance(value, list)
            and len(value) > 0
            and isinstance(value[0], (list, np.ndarray))
        ):
            sg = group.create_group(key)
            sg.attrs["_type"] = _T_RAGGED
            flat = (
                np.concatenate([np.asarray(row, dtype=np.int64) for row in value])
                if any(len(r) > 0 for r in value)
                else np.array([], dtype=np.int64)
            )
            lengths = np.array([len(r) for r in value], dtype=np.int64)
            indptr  = np.concatenate([[0], np.cumsum(lengths)])
            sg.create_dataset("flat",   data=flat)
            sg.create_dataset("indptr", data=indptr)

        # ---- scalar (int, float, str, bool, tuple, list of scalars) ------
        else:
            if isinstance(value, (tuple, list)):
                group.attrs[key] = list(value)
            elif isinstance(value, np.integer):
                group.attrs[key] = int(value)
            elif isinstance(value, np.floating):
                group.attrs[key] = float(value)
            elif isinstance(value, np.bool_):
                group.attrs[key] = bool(value)
            else:
                group.attrs[key] = value


def _write_label_array(group, key, arr):
    """
    Write an object ndarray whose entries are scalars or None as a
    vlen-string HDF5 dataset, recording the element type so it can be
    cast back correctly on load.  None entries are stored as "".

    Parameters
    ----------
    group : h5py.Group
    key : str
    arr : np.ndarray, dtype=object
    """
    import h5py

    non_none  = [v for v in arr if v is not None]
    elem_type = type(non_none[0]).__name__ if non_none else "str"

    str_arr = np.array(
        ["" if v is None else str(v) for v in arr],
        dtype=h5py.special_dtype(vlen=str),
    )
    ds = group.create_dataset(key, data=str_arr)
    ds.attrs["_type"]     = _T_LABEL
    ds.attrs["elem_type"] = elem_type


def _read_dict(group):
    """
    Recursively read an h5py Group written by _write_dict and return a dict
    with values restored to their original Python/numpy types.

    Parameters
    ----------
    group : h5py.Group
        Open HDF5 group previously written by _write_dict.

    Returns
    -------
    dict
    """
    import h5py

    result = {}

    # --- attrs (scalars written directly, no _type tag) -------------------
    for key, value in group.attrs.items():
        if key == "_type":
            continue
        if isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
        elif isinstance(value, np.bool_):
            result[key] = bool(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, bytes):
            result[key] = value.decode()
        else:
            result[key] = value

    # --- datasets and subgroups -------------------------------------------
    for key in group.keys():
        item  = group[key]

        if isinstance(item, h5py.Dataset):
            _type = item.attrs.get("_type", _T_DENSE)
            if _type == _T_DENSE:
                result[key] = item[:]
            elif _type == _T_LABEL:
                result[key] = _read_label_array(item)

        elif isinstance(item, h5py.Group):
            _type = item.attrs.get("_type", _T_DICT)
            if _type == _T_NONE:
                result[key] = None
            elif _type == _T_DICT:
                result[key] = _read_dict(item)
            elif _type == _T_SPARSE:
                result[key] = csr_matrix(
                    (item["data"][:], item["indices"][:], item["indptr"][:]),
                    shape=tuple(item.attrs["shape"]),
                )
            elif _type == _T_RAGGED:
                flat   = item["flat"][:]
                indptr = item["indptr"][:]
                result[key] = [
                    flat[indptr[i]:indptr[i + 1]].tolist()
                    for i in range(len(indptr) - 1)
                ]

    return result


def _read_label_array(dataset):
    """
    Reconstruct an object ndarray from a vlen-string HDF5 dataset written
    by _write_label_array.  Empty strings are restored as None; other
    entries are cast to the recorded elem_type.

    Parameters
    ----------
    dataset : h5py.Dataset

    Returns
    -------
    np.ndarray, dtype=object
    """
    raw       = dataset[:]
    elem_type = str(dataset.attrs.get("elem_type", "str"))
    result    = np.empty(len(raw), dtype=object)

    for i, v in enumerate(raw):
        v_str = v.decode() if isinstance(v, bytes) else str(v)
        if v_str == "":
            result[i] = None
        elif "int" in elem_type:
            result[i] = int(v_str)
        elif "float" in elem_type:
            result[i] = float(v_str)
        else:
            result[i] = v_str

    return result


# ---------------------------------------------------------------------------
# VQRecallerIO
# ---------------------------------------------------------------------------

class VQRecallerIO:
    """
    HDF5 save/load manager for VQRecaller instances.

    Owns the entire HDF5 file structure for standalone recaller files.
    No other module in vqlp imports h5py or knows about the file layout.

    Parameters
    ----------
    recaller : VQRecaller
        The instance to save.  Only required for :meth:`save`; use
        :meth:`load` as a classmethod to reconstruct.

    Examples
    --------
    Save::

        VQRecallerIO(recaller).save("recaller.h5")

    Load::

        recaller = VQRecallerIO.load("recaller.h5")
    """

    _GRP_META     = "meta"
    _GRP_RECALLER = "recaller"

    def __init__(self, recaller):
        self._recaller = recaller

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save the VQRecaller to an HDF5 file at ``path``.

        File layout
        -----------
        ::

            recaller.h5
            ├── meta/          version string and class tag
            └── recaller/      VQRecaller.to_dict() contents

        Parameters
        ----------
        path : str
            Destination file path.  Created or overwritten.
        """
        import h5py
        from importlib.metadata import version as _pkg_version

        try:
            _version = _pkg_version("vqlp")
        except Exception:
            _version = "unknown"

        with h5py.File(path, "w") as f:
            mg = f.create_group(self._GRP_META)
            mg.attrs["vqlp_version"] = _version
            mg.attrs["class"]        = "VQRecaller"

            rg = f.create_group(self._GRP_RECALLER)
            _write_dict(rg, self._recaller.to_dict())

    @classmethod
    def load(cls, path):
        """
        Load a VQRecaller from an HDF5 file written by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.

        Returns
        -------
        VQRecaller
        """
        import h5py
        from vqlp.vq import VQRecaller

        with h5py.File(path, "r") as f:
            d = _read_dict(f[cls._GRP_RECALLER])

        return VQRecaller.from_dict(d)

    # ------------------------------------------------------------------
    # Internal helper called by VQFitterIO (not part of public API)
    # ------------------------------------------------------------------

    @staticmethod
    def write_to_group(group, recaller):
        """Write a VQRecaller into an already-open h5py Group."""
        _write_dict(group, recaller.to_dict())

    @staticmethod
    def read_from_group(group):
        """Reconstruct a VQRecaller from an already-open h5py Group."""
        from vqlp.vq import VQRecaller
        return VQRecaller.from_dict(_read_dict(group))


# ---------------------------------------------------------------------------
# VQFitterIO
# ---------------------------------------------------------------------------

class VQFitterIO:
    """
    HDF5 save/load manager for VQFitter instances.

    Owns the entire HDF5 file structure for fitter files, including the
    embedded VQRecaller.  No other module in vqlp imports h5py or knows
    about the file layout.

    Parameters
    ----------
    fitter : VQFitter
        The instance to save.  Only required for :meth:`save`; use
        :meth:`load` as a classmethod to reconstruct.

    Examples
    --------
    Save::

        VQFitterIO(fitter).save("fitter.h5")

    Load::

        fitter = VQFitterIO.load("fitter.h5")
    """

    _GRP_META     = "meta"
    _GRP_FITTER   = "fitter"
    _GRP_RECALLER = "recaller"

    def __init__(self, fitter):
        self._fitter = fitter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save the VQFitter (and its embedded VQRecaller) to an HDF5 file.

        File layout
        -----------
        ::

            fitter.h5
            ├── meta/          version string and class tag
            ├── fitter/        VQFitter.to_dict() contents (hyperparams + W)
            └── recaller/      VQRecaller.to_dict() contents

        Notes
        -----
        The RNG state (``_rng``) is not saved.  On load the RNG is
        re-initialized from ``random_state``, so post-load fitting calls
        are reproducible from the same seed but do not continue from the
        exact RNG position at save time.

        Parameters
        ----------
        path : str
            Destination file path.  Created or overwritten.
        """
        import h5py
        from importlib.metadata import version as _pkg_version

        try:
            _version = _pkg_version("vqlp")
        except Exception:
            _version = "unknown"

        with h5py.File(path, "w") as f:
            mg = f.create_group(self._GRP_META)
            mg.attrs["vqlp_version"] = _version
            mg.attrs["class"]        = "VQFitter"

            _write_dict(f.create_group(self._GRP_FITTER),
                        self._fitter.to_dict())

            rg = f.create_group(self._GRP_RECALLER)
            VQRecallerIO.write_to_group(rg, self._fitter.recaller)

    @classmethod
    def load(cls, path):
        """
        Load a VQFitter (and its embedded VQRecaller) from an HDF5 file.

        Parameters
        ----------
        path : str
            Path to an HDF5 file previously created by :meth:`save`.

        Returns
        -------
        VQFitter
            Fully reconstructed instance.  The embedded recaller is
            available as ``fitter.recaller`` immediately after load.
        """
        import h5py
        from vqlp.vq import VQFitter

        with h5py.File(path, "r") as f:
            fitter_d   = _read_dict(f[cls._GRP_FITTER])
            recaller   = VQRecallerIO.read_from_group(f[cls._GRP_RECALLER])

        fitter          = VQFitter.from_dict(fitter_d)
        fitter.recaller = recaller
        return fitter