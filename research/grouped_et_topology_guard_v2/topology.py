from __future__ import annotations

import hashlib

import cc3d
import numpy as np
from scipy import ndimage


TOUCH_STRUCTURE = ndimage.generate_binary_structure(3, 2)


def metric_components(
    mask: np.ndarray,
    dilation_iterations: int,
    minimum_prediction_volume: int,
) -> np.ndarray:
    """Replicate the prediction-side component preprocessing in metrics_GLI."""
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.int32)

    dilated = ndimage.binary_dilation(
        binary,
        structure=TOUCH_STRUCTURE,
        iterations=dilation_iterations,
    )
    dilated_components = cc3d.connected_components(
        dilated.astype(np.uint8, copy=False),
        connectivity=26,
    )
    combined = np.where(binary, dilated_components, 0).astype(
        np.int32,
        copy=False,
    )
    counts = np.bincount(combined.ravel())
    small_labels = np.flatnonzero(counts <= minimum_prediction_volume)
    small_labels = small_labels[small_labels != 0]
    if small_labels.size:
        combined[np.isin(combined, small_labels)] = 0
    return combined


def touching_pairs(first: np.ndarray, second: np.ndarray) -> set[tuple[int, int]]:
    """Return the same 18-neighborhood component pairs as metrics_GLI."""
    first_labels = np.unique(first)
    first_labels = first_labels[first_labels != 0]
    pairs: set[tuple[int, int]] = set()
    for first_label in first_labels:
        dilated = ndimage.binary_dilation(
            first == first_label,
            structure=TOUCH_STRUCTURE,
        )
        second_labels = np.unique(second[dilated])
        for second_label in second_labels:
            if second_label != 0:
                pairs.add((int(first_label), int(second_label)))
    return pairs


def relabel_from_pairs(
    labels: np.ndarray,
    pairs: set[tuple[int, int]],
) -> np.ndarray:
    """Replicate relabel_nifti_image without temporary NIfTI files."""
    maximum = int(labels.max(initial=0))
    next_label = maximum + 1
    common_to_new: dict[int, int] = {}
    original_to_new: dict[int, int] = {}
    ordered = sorted(pairs)
    for _, common in ordered:
        if common not in common_to_new:
            common_to_new[common] = next_label
            next_label += 1
    for original, common in ordered:
        if original not in original_to_new:
            original_to_new[original] = common_to_new[common]

    output = labels.copy()
    for original, replacement in original_to_new.items():
        output[labels == original] = replacement
    return output


def reorder_labels(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    unique = unique[unique != 0]
    output = np.zeros(labels.shape, dtype=np.int32)
    for new_label, old_label in enumerate(unique, start=1):
        output[labels == old_label] = new_label
    return output


def rc_topology_labels(segmentation: np.ndarray) -> np.ndarray:
    """Build the official prediction-side RC lesion partition without GT."""
    # SimpleITK exposes arrays as (z, y, x), while metrics_GLI loads them
    # through nibabel as (x, y, z). Component IDs are scan-order dependent,
    # and the vendor relabeling routine uses those IDs when resolving ties.
    segmentation = np.transpose(segmentation, (2, 1, 0))
    et = metric_components(
        segmentation == 3,
        dilation_iterations=3,
        minimum_prediction_volume=10,
    )
    netc = metric_components(
        segmentation == 1,
        dilation_iterations=5,
        minimum_prediction_volume=20,
    )
    rc = metric_components(
        segmentation == 4,
        dilation_iterations=5,
        minimum_prediction_volume=20,
    )

    et_pairs = touching_pairs(rc, et)
    combined = (
        relabel_from_pairs(rc, et_pairs)
        if len(et_pairs) > 1
        else rc.copy()
    )
    netc_pairs = touching_pairs(combined, netc)
    if len(netc_pairs) > 1:
        combined = relabel_from_pairs(combined, netc_pairs)
    return reorder_labels(combined)


def topology_digest(labels: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(labels, dtype=np.int32)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest().upper()
