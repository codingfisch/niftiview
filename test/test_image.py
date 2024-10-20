import os
import unittest
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path

from niftiview.core import PLANES, ATLASES, TEMPLATES
from niftiview.image import NiftiImage
from colorbar.utils import to_numpy
DATA_PATH = f'{Path(__file__).parents[1].resolve()}/test/data'
ATLAS = 'lpba40'
ATLAS_NIB_IMAGE = nib.load(ATLASES[ATLAS])
TEMPLATE_NIB_IMAGES = {template: nib.load(TEMPLATES[template]) for template in ['ch2', 'mni152']}
IMAGE_KWARGS = [{'layout': 's|c/a'}, {'layout': 'a[10]|a[20]|s', 'crosshair': True}, {'aspect_ratios': 1},
                {'coord_sys': 'array_idx'}, {'coord_sys': 'array_mm'}, {'coord_sys': 'scanner_mm'}, {'resizing': 5},
                {'glass_mode': 'max'}, {'transp_if': '<.1'}, {'qrange': (.2, .8)}, {'vrange': (20, 80)},
                {'equal_hist': True}, {'alpha': .8}, {'is_atlas': True}, {'crosshair': True}, {'fpath': True},
                {'coordinates': True}, {'title': 'Title'}, {'header': True}, {'histogram': True}, {'header': True},
                {'histogram': True}, {'cbar': True}, {'tmp_height': 30}, {'cbar': True, 'cbar_pad': 20, 'cbar_x': .1},
                {'cbar': True, 'fontsize': 10, 'linecolor': 'gray', 'linewidth': 9}]


class TestNiftiImage(unittest.TestCase):
    def setUp(self):
        self.kwargs_list = IMAGE_KWARGS

    def test_image(self):
        for template in list(TEMPLATE_NIB_IMAGES):
            for atlas in [False, True]:
                nib_images = [TEMPLATE_NIB_IMAGES[template]]
                if atlas:
                    nib_images.append(ATLAS_NIB_IMAGE)
                nii = NiftiImage(nib_images=nib_images)
                for kwargs in self.kwargs_list:
                    kwarg_string = '_'.join([f'{k}-{v}' for k, v in kwargs.items()])
                    kwarg_string = kwarg_string.replace('/', '_')
                    im = nii.get_image(origin=[8, 4, 4], height=249, **kwargs)
                    fstem = f'{template}_{ATLAS}' if atlas else template
                    filepath = f'{DATA_PATH}/image/{fstem}_{kwarg_string}.png'
                    # im.save(filepath)  # Run with this line before version change
                    im_old = Image.open(filepath)
                    images_equal = np.array_equal(to_numpy(im), to_numpy(im_old))
                    if not images_equal:
                        compare_filepath = f'{DATA_PATH}/{kwarg_string}.png'
                        print(f'Output images changed. Compare new image at {compare_filepath} to old image at {filepath}')
                        im.save(compare_filepath)
                    self.assertTrue(images_equal)

    def test_save(self):
        tmp_filename = f'{DATA_PATH}/tmp'
        for template in list(TEMPLATE_NIB_IMAGES):
            for atlas in [False, True]:
                nib_images = [TEMPLATE_NIB_IMAGES[template]]
                if atlas:
                    nib_images.append(ATLAS_NIB_IMAGE)
                nii = NiftiImage(nib_images=nib_images)
                for kwargs in self.kwargs_list:
                    nii.save_image(f'{tmp_filename}.ps', origin=[8, 4, 4], height=249, **kwargs)
                    tmp = Image.open(f'{tmp_filename}.ps')
                    tmp.save(f'{tmp_filename}.png')
                    tmp.close()
                    im = Image.open(f'{tmp_filename}.png')
                    os.remove(f'{tmp_filename}.ps')
                    os.remove(f'{tmp_filename}.png')
                    kwarg_string = '_'.join([f'{k}-{v}' for k, v in kwargs.items()])
                    kwarg_string = kwarg_string.replace('/', '_')
                    fstem = f'{template}_{ATLAS}' if atlas else template
                    filepath = f'{DATA_PATH}/image_save/{fstem}_{kwarg_string}.png'
                    # im.save(filepath)  # Run with this line before version change
                    im_old = Image.open(filepath)
                    images_equal = np.array_equal(to_numpy(im), to_numpy(im_old))
                    if not images_equal:
                        compare_filepath = f'{DATA_PATH}/{kwarg_string}.png'
                        print(f'Output images changed. Compare new image at {compare_filepath} to old image at {filepath}')
                        im.save(compare_filepath)
                    self.assertTrue(images_equal)


if __name__ == "__main__":
    unittest.main()