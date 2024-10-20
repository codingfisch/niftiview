import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw

from .core import PLANES
from .utils import DATA_PATH
LINE_ARRAYS = {plane: np.load(f'{DATA_PATH}/glassbrain/lines_{plane}.npz') for plane in PLANES}
TEMPLATE = nib.load(f'{DATA_PATH}/niftis/templates/T1.nii.gz')


class GlassBrain:
    def __init__(self):
        self._affine = None
        self._shape = None
        self._image_props = None
        self._linewidth = None
        self._pad_color = None
        self._image = None

    def get_image(self, nifti, linewidth=1, pad_color='white'):
        if self._image is not None:
            same_nifti = (self._shape == nifti.shape and
                          np.array_equal(self._affine, nifti.affine) and
                          equal_image_properties(self._image_props, nifti._image_props))
            if same_nifti and self._linewidth == linewidth and self._pad_color == pad_color:
                return self._image
        lines = self.get_lines(nifti, linewidth)
        im = Image.new('L', nifti.image_size, pad_color)
        draw = ImageDraw.Draw(im)
        for line in lines:
            draw.line(joint='curve', **line)
        self.set_attributes(im, nifti, linewidth, pad_color)
        return im

    def get_lines(self, nifti, linewidth):
        trans = get_translation(nifti, TEMPLATE)
        zoom = get_zoom(nifti, TEMPLATE)
        lines = []
        for plane, box, size in zip(nifti.image_planes, nifti.image_boxes, nifti.image_sizes):
            dims = np.array([i for i in range(3) if i != PLANES.index(plane)])
            plane_lines = self.transform_lines(LINE_ARRAYS[plane], size=size, zoom=zoom[dims],
                                               translation=trans[dims], move=box, width_scale=linewidth)
            lines += plane_lines
        return lines

    def set_attributes(self, im, nifti, linewidth, pad_color):
        self._affine = nifti.affine
        self._shape = nifti.shape
        self._image_props = nifti._image_props
        self._linewidth = linewidth
        self._pad_color = pad_color
        self._image = im

    @staticmethod
    def transform_lines(arrays, size, zoom, translation, move=(0, 0), width_scale=1):
        points, arguments = arrays['points'], arrays['arguments']
        arguments[:, 1] = (width_scale * arguments[:, 1]).round()
        arguments = arguments.astype(int)
        points = np.array(size) * (zoom * points + translation) + np.array(move)
        n = 0
        lines = []
        for args in arguments:
            lines.append({'xy': tuple(map(tuple, points[n:n + args[0]])), 'width': int(args[1]), 'fill': int(args[2])})
            n += args[0]
        return lines


def get_translation(nifti1, nifti2):
    size1 = np.array(nifti1.shape[:3]) * nifti1.affine[np.diag_indices(3)]
    size2 = np.array(nifti2.shape[:3]) * nifti2.affine[np.diag_indices(3)]
    trans1 = size1 + nifti1.affine[:3, 3]
    trans2 = size2 + nifti2.affine[:3, 3]
    return (trans1 - trans2) / size1


def get_zoom(nifti1, nifti2):
    size1 = np.array(nifti1.shape[:3]) * nifti1.affine[np.diag_indices(3)]
    size2 = np.array(nifti2.shape[:3]) * nifti2.affine[np.diag_indices(3)]
    return size2 / size1


def equal_image_properties(props1, props2):
    return [tuple(p.values()) for p in props1] == [tuple(p.values()) for p in props2]
