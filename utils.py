import math
from typing import Any, Optional

import cv2
import numpy as np
from skimage.draw import disk


def get_t(a: int, p: int) -> float:
    return (4 * a) / (math.pi * p ** 2)


def correct_contrast_cpu(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img


def correct_contrast_gpu(img: np.ndarray, stream: Any = None) -> np.ndarray:
    lab = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2LAB, stream=stream)
    l_channel, a, b = cv2.cuda.split(lab, stream=stream)

    clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    cl = clahe.apply(l_channel, stream=stream)

    limg = cv2.cuda.merge((cl, a, b), stream=stream)
    limg_cuda = cv2.cuda.GpuMat()
    limg_cuda.upload(limg, stream=stream)
    enhanced_img = cv2.cuda.cvtColor(limg_cuda, cv2.COLOR_LAB2BGR, stream=stream)

    return enhanced_img


def segment_fovea_cpu(morph_img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_clahe = clahe.apply(morph_img)
    _, img_otsu = cv2.threshold(
        img_clahe,
        thresh=0,
        maxval=255,
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    img_otsu = 255 - img_otsu

    return img_otsu


def reduce_img_noise_cpu(img: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    kernel = disk(9)
    img_top = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    img_bottom = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    morph_img = img_gray + img_top - img_bottom

    return morph_img


def find_best_fovea_region_cpu(img_otsu: np.ndarray,
                               max_candidate_area: int,
                               min_candidate_size: int,
                               min_t: float) -> Optional[tuple[Any, Any]]:
    (nlabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img_otsu, connectivity=4)
    min_area = min_candidate_size * min_candidate_size
    areas = stats[:, cv2.CC_STAT_AREA]
    filtered_labels = np.where((areas > min_area) & (areas < max_candidate_area))

    if len(filtered_labels) == 0:
        return

    label_shapes = []
    for label in filtered_labels[0]:
        arr = np.argwhere(labels == label)

        left = np.min(arr[:, 1])
        right = np.max(arr[:, 1])
        top = np.min(arr[:, 0])
        bottom = np.max(arr[:, 0])

        width = np.abs(left - right)
        height = np.abs(top - bottom)

        area = np.count_nonzero(labels == label)
        p = width if width > height else height

        t = get_t(area, p)

        if t >= min_t:
            t = get_t(area, p)
            label_shapes.append(t)
        else:
            label_shapes.append(0)

    if len(label_shapes) == 0:
        return

    best_shape_id = np.argmax(label_shapes)
    best_centroid_id = filtered_labels[0][best_shape_id]
    best_centroid = centroids[best_centroid_id]
    best_conf = label_shapes[best_shape_id]

    return best_centroid, best_conf


def find_best_disc_region_cpu(img_otsu: np.ndarray,
                              max_candidate_area: int,
                              min_candidate_size: int,
                              min_t: float) -> Optional[tuple[Any, Any]]:
    (nlabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img_otsu, connectivity=4)
    min_area = min_candidate_size * min_candidate_size
    areas = stats[:, cv2.CC_STAT_AREA]
    filtered_labels = np.where((areas > min_area) & (areas < max_candidate_area))

    if len(filtered_labels) == 0:
        return

    label_shapes = []
    for label in filtered_labels[0]:
        arr = np.argwhere(labels == label)

        left = np.min(arr[:, 1])
        right = np.max(arr[:, 1])
        top = np.min(arr[:, 0])
        bottom = np.max(arr[:, 0])

        width = np.abs(left - right)
        height = np.abs(top - bottom)

        area = np.count_nonzero(labels == label)
        p = width if width > height else height

        t = get_t(area, p)

        if t >= min_t:
            label_shapes.append(t)
        else:
            label_shapes.append(0)

    if len(label_shapes) == 0:
        return

    best_shape_id = np.argmax(label_shapes)
    best_centroid_id = filtered_labels[0][best_shape_id]
    best_centroid = centroids[best_centroid_id]
    best_conf = label_shapes[best_shape_id]
    return best_centroid, best_conf


def get_disc_segment(distances: np.ndarray) -> np.ndarray:
    kernel_erode = disk(3)
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    img_erode = cv2.erode(
        distances,
        kernel_erode,
        iterations=1
    )
    img_close = cv2.morphologyEx(
        img_erode,
        cv2.MORPH_CLOSE,
        kernel_close
    )
    img_open = cv2.morphologyEx(
        img_close,
        cv2.MORPH_OPEN,
        kernel_open
    )
    _, img_otsu = cv2.threshold(
        img_open,
        thresh=0,
        maxval=255,
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, hierarchy = cv2.findContours(img_otsu, 1, 2)

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        img_otsu = cv2.drawContours(img_otsu, [hull], -1, 255, thickness=cv2.FILLED)

    return img_otsu


def get_disc_img_tmp_form(distance_margin: int, img: np.ndarray) -> np.ndarray:
    img_gray_0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    img_gray_1 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[..., 1]
    img_gray_3 = img[..., 1]
    img_gray_0_diff = 255 - img_gray_0
    img_gray_1_diff = img_gray_1
    img_gray_3_diff = 255 - img_gray_3
    arr = np.stack((img_gray_0_diff, img_gray_1_diff, img_gray_3_diff), axis=2)
    distances = np.linalg.norm(arr, axis=2)
    min_dist = distances.min()
    distances = np.where(distances <= min_dist + distance_margin, distances, 0).astype(np.uint8)

    return distances


def find_best_fovea_region_gpu(img_otsu: Any,
                               max_candidate_area: int,
                               min_candidate_size: int,
                               min_t: float) -> Any:
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_otsu, connectivity=4)
    min_area = min_candidate_size * min_candidate_size
    areas = stats[:, cv2.CC_STAT_AREA]
    filtered_labels = np.where((areas > min_area) & (areas < max_candidate_area))

    if len(filtered_labels) == 0:
        return

    label_shapes = []
    for label in filtered_labels[0]:
        arr = np.argwhere(labels == label)

        left = np.min(arr[:, 1])
        right = np.max(arr[:, 1])
        top = np.min(arr[:, 0])
        bottom = np.max(arr[:, 0])

        width = np.abs(left - right)
        height = np.abs(top - bottom)

        area = np.count_nonzero(labels == label)
        p = width if width > height else height

        t = get_t(area, p)

        if t >= min_t:
            t = get_t(area, p)
            label_shapes.append(t)
        else:
            label_shapes.append(0)

    if len(label_shapes) == 0:
        return

    best_shape_id = np.argmax(label_shapes)
    best_centroid_id = filtered_labels[0][best_shape_id]
    best_centroid = centroids[best_centroid_id]
    best_conf = label_shapes[best_shape_id]

    return best_centroid, best_conf


def segment_fovea_gpu(img_gray: Any, stream: Any) -> Any:
    kernel = disk(9)

    top_hat_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_TOPHAT, cv2.CV_8U, kernel, (-1, -1), 1)
    black_hat_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_BLACKHAT, cv2.CV_8U, kernel, (-1, -1), 1)
    img_top = top_hat_filter.apply(img_gray, stream=stream)
    img_bottom = black_hat_filter.apply(img_gray, stream=stream)
    morph_img = cv2.cuda.subtract(cv2.cuda.add(img_gray, img_top, stream=stream), img_bottom, stream=stream)
    clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_clahe = clahe.apply(morph_img, stream=stream)
    _, img_otsu = cv2.threshold(
        img_clahe.download(),
        thresh=0,
        maxval=255,
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    img_otsu = 255 - img_otsu

    return img_otsu


def reduce_fovea_noise_gpu(img: Any, stream: Any) -> Any:
    img_gray = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2GRAY, stream=stream)
    gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8U, cv2.CV_8U, (5, 5), 0)
    img_gray = gaussian_filter.apply(img_gray, stream=stream)

    return img_gray


def find_best_disc_region_gpu(img_otsu: Any,
                              max_candidate_area: int,
                              min_candidate_size: int,
                              min_t: float) -> Any:
    (nlabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img_otsu, connectivity=4)
    min_area = min_candidate_size * min_candidate_size
    areas = stats[:, cv2.CC_STAT_AREA]
    filtered_labels = np.where((areas > min_area) & (areas < max_candidate_area))

    if len(filtered_labels) == 0:
        return

    label_shapes = []
    for label in filtered_labels[0]:
        arr = np.argwhere(labels == label)

        left = np.min(arr[:, 1])
        right = np.max(arr[:, 1])
        top = np.min(arr[:, 0])
        bottom = np.max(arr[:, 0])

        width = np.abs(left - right)
        height = np.abs(top - bottom)

        area = np.count_nonzero(labels == label)
        p = width if width > height else height

        t = get_t(area, p)

        if t >= min_t:
            label_shapes.append(t)
        else:
            label_shapes.append(0)

    if len(label_shapes) == 0:
        return

    best_shape_id = np.argmax(label_shapes)
    best_centroid_id = filtered_labels[0][best_shape_id]
    best_centroid = centroids[best_centroid_id]
    best_conf = label_shapes[best_shape_id]

    return best_centroid, best_conf


def get_disc_img_segment_gpu(distances_8u: Any, stream: Any) -> Any:
    kernel_erode = disk(3)
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    erode = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8U, kernel_erode, (-1, -1), 1)
    close = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8U, kernel_close, (-1, -1), 1)
    open = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8U, kernel_open, (-1, -1), 1)
    img_erode = erode.apply(distances_8u, stream=stream)
    img_close = close.apply(img_erode, stream=stream)
    img_open = open.apply(img_close, stream=stream)
    _, img_otsu = cv2.threshold(
        img_open.download(),
        thresh=0,
        maxval=255,
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, hierarchy = cv2.findContours(img_otsu, 1, 2)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        img_otsu = cv2.drawContours(img_otsu, [hull], -1, 255, thickness=cv2.FILLED)

    return img_otsu


def get_disc_img_tmp_form_gpu(distance_margin: int, img: np.ndarray, stream: Any) -> Any:
    img_gray_0, _, _ = cv2.cuda.split(cv2.cuda.cvtColor(img, cv2.COLOR_BGR2LAB, stream=stream), stream=stream)
    _, img_gray_1, _ = cv2.cuda.split(cv2.cuda.cvtColor(img, cv2.COLOR_BGR2YUV, stream=stream), stream=stream)
    _, img_gray_3, _ = cv2.cuda.split(img, stream=stream)
    white_layer_cols, white_layer_rows = img.size()
    white_layer = cv2.cuda.GpuMat(cols=white_layer_cols, rows=white_layer_rows, type=cv2.CV_8UC1)
    white_layer.upload(np.ones((white_layer_rows, white_layer_cols)).astype(np.uint8) + 254, stream=stream)
    img_gray_0_diff = cv2.cuda.subtract(white_layer, img_gray_0, stream=stream)
    img_gray_1 = img_gray_1
    img_gray_3_diff = cv2.cuda.subtract(white_layer, img_gray_3, stream=stream)
    img_gray_0_diff_mat = cv2.cuda.GpuMat(cols=white_layer_cols, rows=white_layer_rows, type=cv2.CV_32F)
    img_gray_1_diff_mat = cv2.cuda.GpuMat(cols=white_layer_cols, rows=white_layer_rows, type=cv2.CV_32F)
    img_gray_3_diff_mat = cv2.cuda.GpuMat(cols=white_layer_cols, rows=white_layer_rows, type=cv2.CV_32F)
    img_gray_0_diff_32f = img_gray_0_diff.convertTo(cv2.CV_32F, img_gray_0_diff_mat)
    img_gray_1_32f = img_gray_1.convertTo(cv2.CV_32F, img_gray_1_diff_mat)
    img_gray_3_diff_32f = img_gray_3_diff.convertTo(cv2.CV_32F, img_gray_3_diff_mat)
    distances = cv2.cuda.sqrt(
        cv2.cuda.add(
            cv2.cuda.add(
                cv2.cuda.sqr(cv2.cuda.abs(img_gray_0_diff_32f, stream=stream), stream=stream),
                cv2.cuda.sqr(cv2.cuda.abs(img_gray_1_32f, stream=stream), stream=stream),
                stream=stream,
            ),
            cv2.cuda.sqr(cv2.cuda.abs(img_gray_3_diff_32f, stream=stream), stream=stream),
            stream=stream,
        ),
        stream=stream
    )
    min_dist, _ = cv2.cuda.minMax(distances)
    _, distances = cv2.cuda.threshold(
        distances,
        thresh=min_dist + distance_margin,
        maxval=10000,
        type=cv2.THRESH_TOZERO_INV,
        stream=stream,
    )
    distances_mat = cv2.cuda.GpuMat(cols=white_layer_cols, rows=white_layer_rows, type=cv2.CV_8U)
    distances_8u = distances.convertTo(cv2.CV_8U, distances_mat)

    return distances_8u
