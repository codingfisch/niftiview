import unittest
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path

from niftiview.core import PLANES, TEMPLATES, NiftiCore
from colorbar.utils import to_numpy
DATA_PATH = f'{Path(__file__).parents[1].resolve()}/test/data'
TEMPLATE_NIB_IMAGES = {template: nib.load(TEMPLATES[template]) for template in ['ch2', 'mni152']}
CORE_KWARGS = [{'layout': 's|c/a'}, {'layout': 'a[10]|a[20]|s'}, {'aspect_ratios': {p: 1 for p in PLANES}},
               {'coord_sys': 'array_idx'}, {'coord_sys': 'array_mm'}, {'coord_sys': 'scanner_mm'}]


class TestNiftiCore(unittest.TestCase):
    def setUp(self):
        self.template = TEMPLATE_NIB_IMAGES['mni152']
        self.max_value = self.template.get_fdata().max()
        self.kwargs_list = CORE_KWARGS

    def test_init(self):
        for template, nib_img in TEMPLATE_NIB_IMAGES.items():
            nib_img = nib.as_closest_canonical(nib_img)
            array = nib_img.get_fdata(dtype=np.float32)

            for nii in [NiftiCore(TEMPLATES[template]), NiftiCore(nib_image=nib_img), NiftiCore(array=array, affine=nib_img.affine)]:
                self.assertTrue(isinstance(nii.array, np.ndarray))
                self.assertTrue(nii.array.ndim == 4)
                self.assertTrue(isinstance(nii.affine, np.ndarray))
                self.assertTrue(nii.affine.shape == (4, 4))

    def test_get_array_slice(self):
        shape = (40, 40, 40, 4)
        array = np.arange(np.prod(shape)).reshape(shape)
        nic = NiftiCore(array=array, affine=np.eye(4))

        self.assertTrue(np.array_equal(array[0, :, :, 0], nic.get_array_slice('sagittal', (0, 0, 0, 0))))
        self.assertTrue(np.array_equal(array[:, 0, :, 1], nic.get_array_slice('coronal', (0, 0, 0, 1))))
        self.assertTrue(np.array_equal(array[:, :, 0, -1], nic.get_array_slice('axial', (0, 0, 0, -1))))

    def test_image(self):
        nic = NiftiCore(nib_image=self.template)
        for kwargs in self.kwargs_list:
            kwarg_string = '_'.join([f'{k}-{v}' for k, v in kwargs.items()])
            kwarg_string = kwarg_string.replace('/', '_')
            im = nic.get_image(origin=[8, 4, 4], height=199, **kwargs)
            im = to_pil_image(im.clip(min=0) / self.max_value)
            filepath = f'{DATA_PATH}/core/{kwarg_string}.png'
            # im.save(filepath)  # Run with this line before version change
            im_old = Image.open(filepath)
            images_equal = np.array_equal(to_numpy(im), to_numpy(im_old))
            if not images_equal:
                compare_filepath = f'{DATA_PATH}/{kwarg_string}.png'
                print(f'Output images changed. Compare new image at {compare_filepath} to old image at {filepath}')
                im.save(compare_filepath)
            self.assertTrue(images_equal)


def to_pil_image(x):
    return Image.fromarray((255 * x).astype(np.uint8))


if __name__ == "__main__":
    unittest.main()
