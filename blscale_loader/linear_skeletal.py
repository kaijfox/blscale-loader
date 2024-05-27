import numpy as np



def construct_transform(skeleton, root_keypt):
    n_kpts = len(skeleton) + 1
    u_to_x = np.zeros([n_kpts, n_kpts])
    x_to_u = np.zeros([n_kpts, n_kpts])
    u_to_x[root_keypt, root_keypt] = 1
    x_to_u[root_keypt, root_keypt] = 1
    # skeleton is topo sorted (thank youuuu)
    for child, parent in skeleton:
        x_to_u[child, parent] = -1
        x_to_u[child, child] = 1
        u_to_x[child] = u_to_x[parent]
        u_to_x[child, child] = 1
    bones_mask = np.ones(n_kpts, dtype=bool)
    bones_mask[root_keypt] = 0
    return {
        "u_to_x": u_to_x,
        "x_to_u": x_to_u,
        "root": root_keypt,
        "bone_mask": bones_mask,
    }


def transform(keypts, transform_data):
    bones_and_root = transform_data["x_to_u"] @ keypts
    return (
        bones_and_root[..., transform_data["root"], :],
        bones_and_root[..., transform_data["bone_mask"], :],
    )


def join_with_root(bones, roots, transform_data):
    return np.insert(bones, transform_data["root"], roots, axis=-2)


def inverse_transform(roots, bones, transform_data):
    if roots is None:
        roots = np.zeros(bones.shape[:-2] + (bones.shape[-1],))
    bones_and_root = join_with_root(bones, roots, transform_data)
    return transform_data["u_to_x"] @ bones_and_root


def roots_and_bones(coords, bones, root_ix):
    """
    Parameters
    ----------
    armature : Armature
        Object with properies bones, keypt_by_name, and root.
    """
    # create skeleton
    ls_mat = construct_transform(bones, root_ix)

    # measure bone lengths by age
    roots_and_bones = {s: transform(coords, ls_mat) for s, coords in coords.items()}
    roots = {s: roots_and_bones[s][0] for s in coords}
    bones = {s: roots_and_bones[s][1] for s in coords}

    return roots, bones, ls_mat


def scale_to_lengths(coords, target_scales, bones, root_ix):
    """
    Parameters
    ----------
    coords : dict of arrays, shape (n_frames, n_keypts, n_dim)
        dictionary of coordinate arrays.
    target_scales : dict of dicts,
        dictionary of target mean inter-keypoint distances and uniform scale
        factors. each sub-dictionary should contain a key `targets` mapping to a
        list of target ikds (or `None` to indicate no scaling), and a key
        `uniform` mapping to a float uniform scale factor.
    config : dict
        kpms config
    """

    roots, bones, ls_mat = roots_and_bones(coords, bones, root_ix)
    n_bones = list(bones.values())[0].shape[-2]
    new_coords = {}

    for s in coords:

        # replace Nones with nans
        tgts = np.atleast_1d(target_scales[s]["targets"])
        tgts = [(np.nan if (t is None) or (t == "None") else t) for t in tgts]
        # expand from scalar / length-1 to full list
        tgts = np.broadcast_to(tgts, [n_bones])

        current_lengths = np.linalg.norm(bones[s], axis=-1).mean(axis=0)
        scale_factors = np.where(np.isfinite(tgts), tgts / current_lengths, 1)
        # apply uniform scale
        scale_factors *= target_scales[s]["uniform"]

        new_bones = bones[s] * scale_factors[None, :, None]
        new_coords[s] = inverse_transform(roots[s], new_bones, ls_mat)

    return new_coords


def reroot(bones, new_root):
    """
    Parameters
    ----------
    bones : np.ndarray
        (n_edges, 2) array of edges.
    root : int
        Index of new root node.

    Returns
    -------
    array
        (n_edges, 2) array of edges, sorted by topological order.
    """
    new_bones = []

    def traverse_from(node):
        visited.add(int(node))
        for child in connected_to(node):
            if int(child) in visited:
                continue
            new_bones.append((child, node))
            traverse_from(child)

    visited = set()

    def connected_to(i):
        x = np.concatenate(
            [
                bones[bones[:, 0] == i, 1],
                bones[bones[:, 1] == i, 0],
            ]
        )
        order = np.argsort(x)
        return x[order]

    traverse_from(new_root)
    return np.array(new_bones)