from typing import Any, Optional

import numpy as np

from msilr_public.utils import correct_contrast_gpu, reduce_fovea_noise_gpu, segment_fovea_gpu, \
    find_best_fovea_region_gpu, get_disc_img_tmp_form_gpu, get_disc_img_segment_gpu, find_best_disc_region_gpu


def get_fovea_center_gpu(img: np.ndarray,
                         min_candidate_size: int = 300,
                         max_candidate_area: int = 300_000,
                         min_t: float = 0.3,
                         stream: Any = None) -> tuple[Optional[list], float]:
    img = correct_contrast_gpu(img, stream=stream)

    img_gray = reduce_fovea_noise_gpu(img, stream)

    img_otsu = segment_fovea_gpu(img_gray, stream)

    best_centroid, best_conf = find_best_fovea_region_gpu(img_otsu, max_candidate_area, min_candidate_size, min_t)

    if not best_centroid or not best_conf:
        return None, 0

    if best_conf > min_t:
        return best_centroid.tolist(), best_conf

    return None, 0


def get_disc_center_gpu(img: np.ndarray,
                        distance_margin: int = 20,
                        min_candidate_size: int = 100,
                        max_candidate_area: int = 300_000,
                        min_t: float = 0.3,
                        stream: Any = None) -> tuple[Optional[list], float]:
    distances_8u = get_disc_img_tmp_form_gpu(distance_margin, img, stream)

    img_otsu = get_disc_img_segment_gpu(distances_8u, stream)

    best_centroid, best_conf = find_best_disc_region_gpu(img_otsu, max_candidate_area, min_candidate_size, min_t)

    if not best_centroid or not best_conf:
        return None, 0

    if best_conf > min_t:
        return best_centroid.tolist(), best_conf

    return None, 0
