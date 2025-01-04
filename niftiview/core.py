import re
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from copy import deepcopy
from colorbar.utils import to_numpy
from affine_image import affine_transform_3d

from .utils import DATA_PATH, get_filestem
PLANES = ('sagittal', 'coronal', 'axial')
PLANE_DICT = {plane[0]: plane for plane in PLANES}
LAYOUT_STRINGS = {'sagittal++': 's|c/a', 'coronal++': 'c|s/a', 'axial++': 'a|c/s', 'all': 'c|s|a',
                  **{plane: plane[0] for plane in PLANES}}
COORDINATE_SYSTEMS = ('array_idx', 'array_mm', 'scanner_mm')
RESIZINGS = ('nearest', 'lanczos', 'bilinear', 'bicubic', 'box', 'hamming')
GLASS_MODES = ('max', 'absmax', 'min')
TEMPLATE_DEFAULT = 'mni152'  # 'ch2' 'T1'
TEMPLATES = {get_filestem(fp): str(fp) for fp in sorted(Path(f'{DATA_PATH}/niftis/templates').glob('*.ni*'))}
ATLASES = {get_filestem(fp): str(fp) for fp in sorted(Path(f'{DATA_PATH}/niftis/atlases').glob('*.ni*'))}


class NiftiCore:
    def __init__(self, filepath=None, nib_image=None, array=None, affine=None, target_affine=None, target_shape=None):
        assert not (array is None and filepath is None and nib_image is None), \
            'Either filepath, nib_image or array must be given'
        if filepath is None and nib_image is None:
            affine = get_dummy_affine(array.shape) if affine is None else affine
            assert isinstance(affine, np.ndarray) and affine.shape == (4, 4), 'Affine must be ndarray with shape (4, 4)'
            array = array.astype(np.float32)
            header = None
        else:
            if filepath is None:
                filepath = nib_image.dataobj.file_like if hasattr(nib_image.dataobj, 'file_like') else None
            array, affine, header = load_np(filepath) if filepath.endswith('.npy') else load_nib(filepath, nib_image)
        assert array.ndim in (3, 4), f'Image shape {array.shape} is not 3D or 4D'
        self.shape = array.shape
        self.sorted_array = None
        self.array = array[..., None] if array.ndim == 3 else array
        self.affine = affine
        self.filepath = filepath
        self.header = header
        if target_shape is not None and target_affine is not None and not np.array_equal(target_affine, affine):
            self.array = resample_3d(self.array, affine, target_affine, target_shape)
            self.affine = target_affine
        self.glass_arrays = self.get_glass_arrays(self.array)
        self.aspect_ratios = get_aspect_ratios(self.affine, self.array.shape)
        self._image_props = []

    @property
    def image_planes(self):
        return [kwargs['plane'] for kwargs in self._image_props]

    @property
    def image_boxes(self):
        return [kwargs['box'] for kwargs in self._image_props]

    @property
    def image_sizes(self):
        return [kwargs['size'] for kwargs in self._image_props]

    @property
    def image_indices(self):
        return [kwargs['idx'] for kwargs in self._image_props]

    @property
    def image_size(self):
        if self._image_props:
            last_plane_props = self._image_props[-1]
            box, size = last_plane_props['box'], last_plane_props['size']
            return box[0] + size[0], box[1] + size[1]
        else:
            return None

    @property
    def dtype(self):
        return self.header.get_data_dtype().name

    @property
    def cal_range(self):
        crange = [0 if self.header is None else self.header['cal_min'].item(),
                  0 if self.header is None else self.header['cal_max'].item()]
        return None if crange == [0, 0] else crange

    def _set_image_properties(self, origin, layout, height, aspect_ratios, coord_sys):
        self._image_props = self.get_image_properties(origin, layout, height, aspect_ratios, coord_sys)

    def quantile(self, q):
        if self.sorted_array is None:
            self.sorted_array = self.sort_array(self.array)
        idxs = (np.array(q) * len(self.sorted_array)).astype(np.int64)
        return self.sorted_array[idxs.clip(min=0, max=len(self.sorted_array) - 1)]

    def quantile_of_value(self, v):
        if self.sorted_array is None:
            self.sorted_array = self.sort_array(self.array)
        return (v > self.sorted_array).mean()

    def get_image(self, origin=None, layout='all', height=400, aspect_ratios=None, coord_sys=None, resizing=1, glass_mode=None):
        self._image_props = self.get_image_properties(origin, layout, height, aspect_ratios, coord_sys)
        im = Image.new('F', self.image_size)
        for kw in self._image_props:
            box_im = self.get_array_slice(kw['plane'], kw['idx'], glass_mode)
            box_im = Image.fromarray(np.rot90(box_im))
            box_im = box_im.resize(kw['size'], resizing) if kw['size'] != box_im.size else box_im
            im.paste(box_im, kw['box'])
        return to_numpy(im)

    def get_image_properties(self, origin, layout, height, aspect_ratios, coord_sys):
        org = np.zeros(4) if origin is None else np.array(list(origin) + [0])[:4]
        layout_string = LAYOUT_STRINGS[layout] if layout in LAYOUT_STRINGS else layout
        aspect_ratios = self.aspect_ratios if aspect_ratios is None else aspect_ratios
        layout_substrings = re.split(r'([\/|])', layout_string)
        image_props = []
        x0 = 0
        for i, substr in enumerate(layout_substrings[::2]):
            props = {'plane': PLANE_DICT[substr[0]]}
            dim = PLANES.index(props['plane'])
            plane_org = org.copy()
            plane_org[dim] = int(substr[2:-1]) if '[' in substr else plane_org[dim]
            props.update({'idx': get_array_index(plane_org, self.affine, self.array.shape, coord_sys)})
            three_planes_substr = (['s', '|'] + layout_substrings + ['s', '|'])[2*i:2*i+5]
            size = self.get_plane_size(height, aspect_ratios, props['plane'], three_planes_substr)
            props.update({'size': size})
            is_bottom_box = three_planes_substr[1] == '/'
            props.update({'box': (x0, height - size[1] if is_bottom_box else 0)})
            x0 += size[0] if size[1] == height or is_bottom_box else 0
            image_props.append(props)
        return tuple(image_props)

    def get_array_slice(self, plane, org_idx, glass_mode=None):
        assert glass_mode is None or glass_mode in GLASS_MODES, 'glass_mode has to be None or in GLASS_MODES'
        if glass_mode is None:
            slice_ = [slice(None), slice(None), slice(None), org_idx[-1]]
            slice_[PLANES.index(plane)] = org_idx[PLANES.index(plane)]
            return self.array[tuple(slice_)]
        else:
            return self.glass_arrays[glass_mode][plane]

    def get_origin_bounds(self, coord_sys=None):
        coord_sys = coord_sys or COORDINATE_SYSTEMS[0]
        assert coord_sys in COORDINATE_SYSTEMS, f'origin_unit must be in {COORDINATE_SYSTEMS}'
        inv_affine = np.linalg.inv(self.affine)[:3]
        if coord_sys == 'scanner_mm':
            lower = -inv_affine[:, 3] / inv_affine.diagonal()
            upper = np.array(self.shape[:3]) - inv_affine[:, 3] / inv_affine.diagonal()
        else:
            lower = -inv_affine[:, 3]
            upper = np.array(self.shape[:3]) - inv_affine[:, 3]
            if coord_sys == 'array_mm':
                lower *= self.affine[:3, :3].sum(axis=1)
                upper *= self.affine[:3, :3].sum(axis=1)
        return np.array([lower, upper])

    @staticmethod
    def get_plane_size(height, aspect_ratios, plane, three_planes_substring):
        aspect_ratios = aspect_ratios if isinstance(aspect_ratios, dict) else {plane: aspect_ratios for plane in PLANES}
        if min(list(aspect_ratios.values())) < .1 or max(list(aspect_ratios.values())) > 10:
            aspect_ratios = {plane: 1 for plane in PLANES}
        if '/' in three_planes_substring:
            idx = three_planes_substring.index('/')
            planes = [PLANE_DICT[three_planes_substring[idx - 1][0]],
                      PLANE_DICT[three_planes_substring[idx + 1][0]]]
            aspect_ratios = [aspect_ratios[planes[0]], aspect_ratios[planes[1]]]
            width = height * aspect_ratios[0] * aspect_ratios[1] / sum(aspect_ratios)
            height = width / aspect_ratios[0 if idx == 3 else 1]
        else:
            width = height * aspect_ratios[plane]
        return int(round(width)), int(round(height))

    @staticmethod
    def get_glass_arrays(array):
        return {'max': {plane: array[..., 0].max(i) for i, plane in enumerate(PLANES)},
                'absmax': {plane: np.abs(array[..., 0]).max(i) for i, plane in enumerate(PLANES)},
                'min': {plane: array[..., 0].min(i) for i, plane in enumerate(PLANES)}}

    @staticmethod
    def sort_array(array):
        return np.sort(array.flatten(order='F'))


def load_nib(filepath=None, nib_image=None):
    nib_img = nib.load(filepath) if nib_image is None else deepcopy(nib_image)
    nib_img = nib.as_closest_canonical(nib_img)
    array = nib_img.get_fdata(dtype=np.float32)
    array = np.nan_to_num(array)
    return array, nib_img.affine, nib_img.header


def load_np(filepath):
    array = np.load(filepath)
    affine = get_dummy_affine(array.shape)
    header = None
    return array, affine, header


def get_dummy_affine(shape):
    affine = np.eye(4)
    affine[:3, 3] -= np.array(shape[:3]) / 2
    return affine


def get_aspect_ratios(affine, shape):
    size = np.diag(affine)[:3] * shape[:3]
    return {PLANES[0]: size[1] / size[2], PLANES[1]: size[0] / size[2], PLANES[2]: size[0] / size[1]}


def resample_3d(x, affine0, affine1, shape, nearest=True):
    affine = np.linalg.inv(affine0) @ affine1
    x_resampled = affine_transform_3d(x.transpose(3, 0, 1, 2)[None], affine[None], shape, nearest, scipy_affine=True)
    return x_resampled[0].transpose(1, 2, 3, 0)


def get_array_index(origin, affine, shape, coord_sys=None):
    coord_sys = coord_sys or COORDINATE_SYSTEMS[0]
    assert coord_sys in COORDINATE_SYSTEMS, f'origin_unit must be in {COORDINATE_SYSTEMS}'
    org = origin[:3].astype(float)
    inv_affine = np.linalg.inv(affine)
    if coord_sys == 'scanner_mm':
        org = (inv_affine @ np.append(org, 1))[:3]
    else:
        if coord_sys == 'array_mm':
            org = org / affine[:3, :3].sum(axis=1)
        org += inv_affine[:3, 3]
    org = np.concatenate([org, origin[3:]])
    org = org.clip(min=0, max=np.array(shape) - 1)
    return tuple(np.round(org).astype(int))
