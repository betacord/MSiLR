from typing import Optional

import numpy as np

from msilr_public.utils import correct_contrast_cpu, get_t, reduce_img_noise_cpu, segment_fovea_cpu, \
    find_best_fovea_region_cpu, get_disc_img_tmp_form, get_disc_segment, find_best_disc_region_cpu


def get_fovea_center_cpu(img: np.ndarray,
                         min_candidate_size: int = 300,
                         max_candidate_area: int = 300_000,
                         min_t: float = 0.3) -> tuple[Optional[list], float]:
    img = correct_contrast_cpu(img)

    morph_img = reduce_img_noise_cpu(img)

    img_otsu = segment_fovea_cpu(morph_img)

    best_centroid, best_conf = find_best_fovea_region_cpu(img_otsu, max_candidate_area, min_candidate_size, min_t)

    if not best_centroid or not best_conf:
        return None, 0

    if best_conf > min_t:
        return best_centroid.tolist(), best_conf

    return None, 0


def get_disc_center_cpu(img: np.ndarray,
                        distance_margin: int = 20,
                        min_candidate_size: int = 100,
                        max_candidate_area: int = 300_000,
                        min_t: float = 0.3) -> tuple[Optional[list], float]:
    distances = get_disc_img_tmp_form(distance_margin, img)

    img_otsu = get_disc_segment(distances)

    best_centroid, best_conf = find_best_disc_region_cpu(img_otsu, max_candidate_area, min_candidate_size, min_t)

    if not best_centroid or not best_conf:
        return None, 0

    if best_conf > min_t:
        return best_centroid.tolist(), best_conf

    return None, 0
