from textwrap import fill
import joblib as jl
import numpy as np
import fnmatch
import tqdm
import os.path
import re, glob
from collections import defaultdict


modata_name_func = lambda path, *a: re.search(
    r"(?:/.*)+/\d{2}_\d{2}_\d{2}_(\d+wk_m\d+)\.gimbal_results\.p", path
).group(1)
modata_name_func_ext = lambda ext: (
    lambda path, *a: re.search(
        r"(?:/.*)+/\d{2}_\d{2}_\d{2}_(\d+wk_m\d+)" + re.escape(ext), path
    ).group(1)
)
modata_age_from_sess_name = lambda name: name.split("-")[-1].split("w")[0]  # 4wk_m0
modata_id_from_sess_name = lambda name: name.split("-")[-1].split("m")[1]


def _name_from_path(filepath, path_in_name, path_sep, remove_extension):
    """Create a name from a filepath.

    Either return the name of the file (with the extension removed) or return
    the full filepath, where the path separators are replaced with `path_sep`.
    """
    if remove_extension:
        filepath = os.path.splitext(filepath)[0]
    if path_in_name:
        return filepath.replace(os.path.sep, path_sep)
    else:
        return os.path.basename(filepath)


def load_glob(
    filepath_pattern,
    loader,
    recursive=True,
    path_sep="-",
    path_in_name=False,
    remove_extension=True,
    name_func=None,
):
    """
    Returns
    -------
    coordinates: dict
        Dictionary mapping filenames to keypoint coordinates as ndarrays of
        shape (n_frames, n_bodyparts, 2[or 3])

    confidences: dict
        Dictionary mapping filenames to `likelihood` scores as ndarrays of
        shape (n_frames, n_bodyparts)
    bodyparts: list of str
        List of bodypart names. The order of the names matches the order of the
        bodyparts in `coordinates` and `confidences`.
    """

    import glob

    filepaths = glob.glob(filepath_pattern, recursive=recursive)
    assert len(filepaths) > 0, fill(f"No such files {filepath_pattern}")

    return load_keypoints(
        filepaths, loader, path_sep, path_in_name, remove_extension, name_func
    )


def load_keypoints(
    filepaths,
    loader,
    path_sep="-",
    path_in_name=False,
    remove_extension=True,
    name_func=None,
):
    coordinates, confidences = {}, {}
    for filepath in tqdm.tqdm(filepaths, desc=f"Loading keypoints", ncols=72):
        try:
            name = (_name_from_path if name_func is None else name_func)(
                filepath, path_in_name, path_sep, remove_extension
            )
            new_coordinates, new_confidences = loader(filepath, name)

            if set(new_coordinates.keys()) & set(coordinates.keys()):
                raise ValueError(
                    f"Duplicate names found in filename list:\n\n"
                    f"{set(new_coordinates.keys()) & set(coordinates.keys())}"
                    f"\n\nThis may be caused by repeated filenames with "
                    "different extensions. If so, please set the extension "
                    "explicitly via the `extension` argument. Another possible"
                    " cause is commonly-named files in different directories. "
                    "if that is the case, then set `path_in_name=True`."
                )

        except Exception as e:
            raise e
            print(fill(f"Error loading {filepath}: {e}"))

        coordinates.update(new_coordinates)
        confidences.update(new_confidences)

    assert len(coordinates) > 0, fill(f"No valid results found")

    return coordinates, confidences


def create_multicam_gimbal_loader():
    def multicam_gimbal_loader(filepath, name):
        # (n_frames, n_bodyparts, n_dim)
        coords = jl.load(filepath)["positions_medfilter"]
        confs = np.ones_like(coords[..., 0])
        return {name: coords}, {name: confs}

    return multicam_gimbal_loader


def create_multicam_npy_loader():
    def multicam_npy_loader(filepath, name):
        # (n_frames, n_bodyparts, n_dim)
        coords = np.load(filepath)
        confs = np.ones_like(coords[..., 0])
        return {name: coords}, {name: confs}

    return multicam_npy_loader


# group grabbers
# -----------------------------------------------------------------------------


def with_age(data_dir, tgt_age, config):
    filenames = glob.glob(f"{data_dir}/**/*.gimbal_results.p", recursive=True)
    sessions = [
        modata_name_func(f)
        for f in filenames
        if modata_age_from_sess_name(modata_name_func(f)) == tgt_age
    ]
    filenames = [f for f in filenames if modata_name_func(f) in sessions]

    coordinates, confidences, _ = load_keypoints(
        filenames,
        create_multicam_gimbal_loader(config["bodyparts"]),
        name_func=modata_name_func,
    )

    return sessions, (coordinates, confidences)


def _create_loader(extension):
    if extension == ".gimbal_results.p":
        loader = create_multicam_gimbal_loader()
    elif extension == ".npy":
        loader = create_multicam_npy_loader()
    else:
        raise ValueError(f"unknown extension {extension}")
    return loader

def by_age(
    data_dir,
    config,
    name_func=modata_name_func,
    age_func=None,
    extension=".gimbal_results.p",
    name_whitelist=None,
    exclude=(),
):
    if age_func is None:
        age_func = modata_age_from_sess_name

    filenames = glob.glob(f"{data_dir}/**/*{extension}", recursive=True)
    groups = defaultdict(list)
    sessions = defaultdict(list)
    for f in filenames:
        name = name_func(f)
        age = age_func(name)
        if age not in exclude and (name_whitelist is None or name in name_whitelist):
            groups[age].append(f)
            sessions[age].append(name)

    loader = _create_loader(extension, config)

    coords, confs = {}, {}
    for age, filenames in groups.items():
        coords[age], confs[age], _ = load_keypoints(
            filenames, loader, name_func=name_func
        )

    return sessions, (coords, confs)


def from_sources_dict(
    data_dir,
    sources_dict,
    name_func=modata_name_func,
    extension=".gimbal_results.p",
):
    """
    Parameters
    ----------
    data_dir : str or Pathlike
        Path to directory containing source data files.
    sources_dict : dict
        Mapping session names (as they should be named in output) to names as
        returned by `name_func`, giving the session data that should be assigned
        each session name.
    config : dict
        KPMS config
    name_func : Callable
        Mapping file paths to session names (in this case, names of source data
        sessions that will be possibly copies and renamed according to
        sources_dict).
    extension : str
        File extension used to glob for source data files in `data_dir`.
    """
    # identify file paths of source sessions
    filenames = glob.glob(f"{data_dir}/**/*{extension}", recursive=True)
    source_sessions = {name_func(f): f for f in filenames}
    
    # check that all source sessions are present and limit filenames to sessions
    # needed to create the dataset
    not_present = set([
        src_n for src_n in sources_dict.values()
        if src_n not in source_sessions
    ])
    if len(not_present):
        err_str = '\', \''.join(not_present)
        raise ValueError(
            f"Source session(s) not found in data directory: '{err_str}'"
        )
    load_filenames = [source_sessions[src_n] for src_n in set(sources_dict.values())]

    # load keypoints
    loader = _create_loader(extension)
    source_coords, source_confs = {}, {}
    source_coords, source_confs = load_keypoints(
        load_filenames, loader, name_func=name_func
    )
    
    # map source data to target session names
    coords = {n: source_coords[src_n] for n, src_n in sources_dict.items()}
    confs = {n: source_confs[src_n] for n, src_n in sources_dict.items()}

    return coords, confs


def expand_pattern_keys(library, mapping):
    """
    Parameters
    ----------
    library : list
        Valid keys.
    mapping : dict
        Mapping whose keys are fnmatch-style patterns to be matched to elements
        of the library."""
    new_mapping = {}
    matched = defaultdict(list)
    for pattern, v in mapping.items():
        matches = fnmatch.filter(library, pattern)
        for lib_item in matches:
            new_mapping[lib_item] = v
            matched[lib_item].append(pattern)
    for lib_item, patterns in matched.items():
        if len(patterns) > 1:
            print(f"Warning: multiple matches found for key `{lib_item}`:", patterns)
    return new_mapping



def merge_dicts(*ds):
    """merge(a, b, ...) = {**a, **b, ...}"""
    ret = {}
    for d in ds:
        ret.update(d)
    return ret


# keypt_io.py
# -----------------------------------------------------------------------------


def get_groups_dict(metadata_val):
    """
    Parameters
    ----------
    metadata_val : dict
        Mapping session names to metadata values
    """
    groups = {}
    for sess_name, val in metadata_val.items():
        if val not in groups:
            groups[val] = []
        groups[val].append(sess_name)
    sorted_keys = sorted(groups.keys())
    return sorted_keys, tuple(groups[k] for k in sorted_keys)
