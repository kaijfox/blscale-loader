import os
import joblib as jl
import numpy as np
from ruamel.yaml import YAML
from .io import from_sources_dict, expand_pattern_keys, modata_name_func_ext
from .linear_skeletal import scale_to_lengths


def external_scale(
    data_dir,
    sources,
    scales,
    bones,
    root_keypoint_ix,
    ext=".gimbal_results.p",
    name_func=None,
):
    """
    Using a

    Most params are as for `modata_with_exclusions`.
    Parameters
    ----------
    sources : str
        Path to a yaml file session names for the dataset to names (as produced
        by `name_func`) of files whose data they should contain. See
        `kpms_custom_io.from_sources_dict`.
    scales : str
        Path to a yaml file containing a dict mapping names (keys of the sources
        file) to an array of target mean inter-keypoint distances and uniform
        scales. See linear_skeletal.scale_to_lengths.
    """

    sources, scales = _align_scales_sources(sources, scales, ext, name_func)
    coords, confs = from_sources_dict(
        data_dir,
        sources,
        extension=ext,
        name_func=name_func,
    )
    coords = scale_to_lengths(coords, scales, bones, root_keypoint_ix)

    print(f"Sessions: {list(coords.keys())}")
    return coords, confs, {}


def _align_scales_sources(sources, scales, ext, name_func = None):
    if isinstance(sources, (str, os.PathLike)):
        sources = YAML().load(sources)
    if isinstance(scales, (str, os.PathLike)):
        scales = YAML().load(scales)
    if name_func is None:
        name_func = modata_name_func_ext(ext)

    scales = expand_pattern_keys(sources.keys(), scales)
    sources = {s: t for s, t in sources.items() if s in scales}

    return sources, scales

def find_sessions(
    sources,
    scales,
    ext=".gimbal_results.p",
    name_func=None,
):
    """Return names of the sessions that will be loaded by `external_scale`."""

    sources, _ = _align_scales_sources(sources, scales, ext, name_func)
    return list(sources.keys())