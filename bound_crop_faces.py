from align_dlib import *
from pathlib import Path

align_dlib_model = AlignDlib(str(Path("./shape_predictor_68_face_landmarks.dat").absolute()))


def preprocess_image(input_path, output_path, crop_dim):
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """
    image = _process_image(input_path, crop_dim)
    if image is not None:
        print(f'Writing processed file: {output_path}')
        cv2.imwrite(output_path, image)

def _process_image(filename, crop_dim):
    image = None
    aligned_image = None

    image = _buffer_image(filename)

    if image is not None:
        aligned_image = _align_image(image, crop_dim)
    else:
        raise IOError(f'Error buffering image: {filename}')

    return aligned_image


def _buffer_image(filename):
    print(f'Reading image: {filename}')
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _align_image(image, crop_dim):
    bb = align_dlib_model.getLargestFaceBoundingBox(image)
    aligned = align_dlib_model.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned


import glob
images = glob.glob('./real_faces/images1024x1024/*.png')

for image in images:
    out = './cropped_real_faces/' + image.split('/')[3]
    print(out)
    preprocess_image(image, out, 256)