import cv2

from msilr_public.cpu import get_fovea_center_cpu, get_disc_center_cpu
from msilr_public.gpu import get_fovea_center_gpu, get_disc_center_gpu

if __name__ == '__main__':
    img_path = '/path/to/file.ext'
    input_img = cv2.imread(img_path)
    mode_no = 0

    get_fovea_center = get_fovea_center_cpu
    get_disc_center = get_disc_center_cpu

    if mode_no == 1:
        get_fovea_center = get_fovea_center_gpu
        get_disc_center = get_disc_center_gpu

    fovea_center = get_fovea_center(input_img)
    disc_center = get_disc_center(input_img)

    print(f'Fovea center: {fovea_center}, disc center: {disc_center}')
