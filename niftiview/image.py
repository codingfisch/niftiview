import numpy as np
from PIL import Image
from cmap import Colormap
from colorbar.utils import to_numpy
from nibabel.filebasedimages import FileBasedImage

from .glass import GlassBrain
from .core import NiftiCore
from .overlay import Overlay
QRANGE = ((0., .99), (0., 1.))
CMAPS_IMAGE = ('gray', 'binary', 'nipy_spectral', 'stern', 'HiLo')
CMAPS_MASK = ('glasbey', 'reds', 'blues', 'set1', 'hot')
CMAP = CMAPS_IMAGE[0]
CMAP_MASK = CMAPS_MASK[0]


class NiftiImage:
    def __init__(self, filepaths=None, nib_images=None, arrays=None, affines=None):
        self.nics = get_nifti_cores(filepaths, nib_images, arrays, affines)
        self.glassbrain = GlassBrain()
        self.cmaps = None
        self.image = None
        self.overlay = None

    def __len__(self):
        return len(self.nics)

    def save_image(self, filepath, origin=(0, 0, 0), layout='all', height=400, aspect_ratios=None, coord_sys=None,
                   resizing=None, glass_mode=None, cmap=None, transp_if=None, qrange=None, vrange=None,
                   equal_hist=False, is_atlas=False, alpha=.5, crosshair=False, fpath=False, coordinates=False,
                   header=False, histogram=False, cbar=False, title=None, fontsize=20, linecolor='w', linewidth=2,
                   extra_lines=None, extra_texts=None, **cbar_kwargs):
        self.overlay = None
        im = self.get_image(origin, layout, height, aspect_ratios, coord_sys, resizing, glass_mode,
                            cmap, transp_if, qrange, vrange, equal_hist, is_atlas, alpha)
        overlay = Overlay(self.nics[-1:], self.cmaps[-1], self.cmaps[-1].vrange[0], self.cmaps[-1].vrange[-1],
                          extra_lines=extra_lines, extra_texts=extra_texts)
        overlay.save(filepath, im, crosshair, fpath, coordinates, header, histogram,
                     cbar, title, fontsize, linecolor, linewidth, **cbar_kwargs)

    def get_image(self, origin=(0, 0, 0), layout='all', height=400, aspect_ratios=None, coord_sys=None, resizing=None,
                  glass_mode=None, cmap=None, transp_if=None, qrange=None, vrange=None, equal_hist=False,
                  is_atlas=False, alpha=.5, crosshair=False, fpath=False, coordinates=False, header=False,
                  histogram=False, cbar=False, title=None, fontsize=20, linecolor='w', linewidth=2, tmp_height=None,
                  as_array=False, **cbar_kwargs):
        if cbar and 'cbar_vertical' in cbar_kwargs and 'cbar_pad' in cbar_kwargs:
            height = height if cbar_kwargs['cbar_vertical'] else height - cbar_kwargs['cbar_pad']
        force_rgba = cbar and 'cbar_pad_color' not in ['black', 'white', 'gray', 'k', 'w']
        layer_height = height if tmp_height is None else height if height <= tmp_height else tmp_height
        layers = self.get_image_layers(origin, layout, layer_height, aspect_ratios, coord_sys, resizing, glass_mode,
                                       cmap, transp_if, qrange, vrange, equal_hist, is_atlas, linewidth, force_rgba)
        self.image = blend_image_layers(layers, alpha)
        if layer_height != height:
            for i in range(len(self.nics)):
                self.nics[i]._set_image_properties(origin, layout, height, aspect_ratios, coord_sys)
            self.image = self.image.resize(size=self.nics[0].image_size, resample=0)
        im = self.image.copy()
        if crosshair or fpath or coordinates or header or histogram or cbar or title is not None:
            self.overlay = Overlay(self.nics, self.cmaps[-1], self.cmaps[-1].vrange[0], self.cmaps[-1].vrange[-1])
            im = self.overlay.draw(im, crosshair, fpath, coordinates, header, histogram, cbar, title,
                                   fontsize, linecolor, linewidth, **cbar_kwargs)
        return to_numpy(im) if as_array else im

    def get_image_layers(self, origin, layout, height, aspect_ratios, coord_sys, resizing, glass_mode,
                         cmap, transp_if, qrange, vrange, equal_hist, is_atlas, linewidth, force_rgba):
        self.cmaps, layers = [], []
        for i, nic in enumerate(self.nics):
            resize_mode = i == 0 if resizing is None else resizing if isinstance(resizing, int) else resizing[i]
            im = nic.get_image(origin, layout, height, aspect_ratios, coord_sys, resize_mode, glass_mode)
            colormap = self.get_cmap(nic, i, cmap, transp_if, qrange, vrange, equal_hist, is_atlas, force_rgba)
            layers.append(colormap(im, asarray=False))
            self.cmaps.append(colormap)
        if glass_mode is not None:
            layers[0] = self.glassbrain.get_image(self.nics[0], linewidth=linewidth)
        return layers

    def get_cmap(self, nic, idx, cmap, transp_if, qrange, vrange, equal_hist, is_atlas, force_rgba):
        cmap_str = cmap or [CMAP, CMAP_MASK][bool(idx)] if isinstance(cmap, (str, type(None))) else cmap[idx]
        value_range = self.get_value_range(nic, idx, vrange, qrange, equal_hist)
        is_an_atlas = is_atlas if isinstance(is_atlas, bool) else is_atlas[idx]
        transparent_if = [None, '=0'][bool(idx)] if transp_if is None else transp_if if isinstance(transp_if, str) else transp_if[idx]
        equalize_histogram = equal_hist and idx == 0
        return TransparentColormap(cmap_str, value_range, is_an_atlas, transparent_if, equalize_histogram, force_rgba)

    @staticmethod
    def get_value_range(nic, idx, vrange, qrange, equal_hist):
        if vrange is None:
            value_range = None
        else:
            value_range = vrange[idx] if hasattr(vrange[0], '__iter__') or vrange[0] is None else vrange
        qrange_is_none = qrange[idx] is None if hasattr(qrange, '__iter__') else qrange is None
        value_range = nic.cal_range if qrange_is_none and value_range is None else value_range
        quantile_range = QRANGE[min(1, idx)] if qrange_is_none else qrange if isinstance(qrange[0], float) else qrange[idx]
        if equal_hist and idx == 0:
            if value_range is not None:
                quantile_range = (nic.quantile_of_value(value_range[0]), nic.quantile_of_value(value_range[-1]))
            return nic.quantile(np.linspace(*quantile_range, 100))
        else:
            value_range = nic.quantile(quantile_range) if value_range is None else value_range
            value_range = nic.quantile((0, 1)) if value_range[0] == value_range[-1] else value_range
            if value_range is not None:
                if value_range[0] == value_range[1]:
                    value_range = np.array([0, value_range[0]] if value_range[0] > 0 else [value_range[0], 0])
            return value_range


class TransparentColormap:
    def __init__(self, cmap='gray', vrange=None, is_atlas=False, transp_if=None, equal_hist=False, force_rgba=False,
                 gamma=1.):
        colormap = Colormap(cmap)
        lut = colormap.lut(gamma=gamma)
        self.name = colormap.name
        self.is_atlas = is_atlas
        self.transp_if = transp_if
        self.vrange = [0, len(lut) if len(lut) < 257 or 'glasbey' in self.name else 1] if is_atlas else vrange
        self.lut = (lut * 255).astype(np.uint8)
        self.equal_hist = equal_hist
        self.force_rgba = force_rgba or transp_if is not None

    def __call__(self, x, asarray=True):
        if self.transp_if is not None:
            op, v = self.transp_if[0], float(self.transp_if[1:])
            mask = np.abs(x) < v if op == '|' else x < v if op == '<' else x == v
            mask = Image.fromarray(255 * (~mask).astype(np.uint8))
        x = normalize(x, self.vrange, self.equal_hist)#, clip=False)
        if not self.force_rgba and self.name in ['matlab:gray', 'matplotlib:binary']:
            x = (255 * x).astype(np.uint8)
            x = x if self.lut[0, 0] == 0 else 255 - x
            x = Image.fromarray(x)
        else:
            x *= len(self.lut) - 1
            x = Image.fromarray(x.astype(np.uint8))
            x.putpalette(self.lut, rawmode='RGBA')
            x = x.convert('RGBA')
        if self.transp_if is not None:
            x.putalpha(mask)
        return to_numpy(x) if asarray else x


def get_nifti_cores(filepaths=None, nib_images=None, arrays=None, affines=None):
    assert not (filepaths is None and nib_images is None and arrays is None), 'Either filepaths, nib_images or arrays must be given'
    filepaths = [filepaths] if isinstance(filepaths, str) else filepaths
    nib_images = [nib_images] if isinstance(nib_images, FileBasedImage) else nib_images
    arrays = [arrays] if isinstance(arrays, np.ndarray) else arrays
    affines = [affines] if isinstance(affines, np.ndarray) else affines
    nics = []
    for i in range(len(filepaths if filepaths is not None else nib_images if nib_images is not None else arrays)):
        nic = NiftiCore(None if filepaths is None else filepaths[i], None if nib_images is None else nib_images[i],
                        None if arrays is None else arrays[i], None if affines is None else affines[i],
                        target_affine=nics[0].affine if i else None, target_shape=nics[0].shape[:3] if i else None)
        nics.append(nic)
    return nics


def blend_image_layers(layers, alpha):
    if len(layers) == 1:
        return layers[0]
    else:
        layers = [im if im.mode == 'RGBA' else im.convert('RGBA') for im in layers]
        mask = layers[1]
        mask.putalpha(mask.split()[3].point(lambda i: i * alpha))
        for im in layers[2:]:
            im.putalpha(im.split()[3].point(lambda i: i * alpha))
            mask = Image.alpha_composite(mask, im)
        return Image.alpha_composite(layers[0], mask)


def normalize(x, vrange, equal_hist=False, clip=True):
    if equal_hist:
        x = np.interp(x, vrange, fp=np.linspace(0., 1., len(vrange), dtype=np.float32))
    else:
        x = (x - vrange[0]) / (vrange[-1] - vrange[0])
    return x.clip(min=0, max=1) if clip else x
