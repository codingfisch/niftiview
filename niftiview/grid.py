import numpy as np
from PIL import Image
from cairo import Context
from tqdm.contrib.concurrent import thread_map
from colorbar.utils import to_numpy, get_color_values

from .core import PLANES, LAYOUT_STRINGS
from .image import NiftiImage
from .overlay import from_pil, get_pad_box, draw_background, get_surface_class


class NiftiImageGrid:
    def __init__(self, filepaths=None, nib_images=None, arrays=None, affines=None):
        assert not (filepaths is None and nib_images is None and arrays is None), 'Either filepaths, nib_images or arrays must be given'
        filepaths = [filepaths] if isinstance(filepaths, str) else filepaths
        n = len(filepaths if filepaths is not None else nib_images if nib_images is not None else arrays)
        filepaths = [None] * n if filepaths is None else filepaths
        nib_images = [None] * n if nib_images is None else nib_images
        arrays = [None] * n if arrays is None else arrays
        affines = [None] * n if affines is None else affines
        self.niis = thread_map(lambda args: NiftiImage(*args), zip(filepaths, nib_images, arrays, affines),
                               disable=True, max_workers=4)
        self.shape = None
        self.boxes = None
        self.patches = None

    def __len__(self):
        return len(self.niis)

    def save_image(self, filepath, origin=(0, 0, 0), layout='all', height=400, squeeze=False, coord_sys=None,
                   resizing=None, glass_mode=None, cmap=None, transp_if=None, qrange=None, vrange=None,
                   equal_hist=False, is_atlas=False, alpha=.5, crosshair=False, fpath=False, coordinates=False,
                   header=False, histogram=False, cbar=False, title=None, fontsize=20, linecolor='w', linewidth=2,
                   tmp_height=None, nrows=None, **cbar_kwargs):
        for i in range(len(self.niis)):
            self.niis[i].overlay = None
        im = self.get_image(origin, layout, height, squeeze, coord_sys, resizing, glass_mode, cmap, transp_if, qrange,
                            vrange, equal_hist, is_atlas, alpha, crosshair, fpath, coordinates, header, histogram,
                            cbar, title, fontsize, linecolor, linewidth, tmp_height, nrows, **cbar_kwargs)
        SurfaceClass = get_surface_class(filepath.split('.')[-1])
        with SurfaceClass(filepath, im.size[0], im.size[1]) as surface:
            ctx = Context(surface)
            ctx = draw_background(ctx, **cbar_kwargs)
            for nii, box in zip(self.niis, self.boxes):
                pad_box = get_pad_box(**cbar_kwargs)
                pad_box = (box[0] + pad_box[0], box[1] + pad_box[1])
                ctx.set_source_surface(from_pil(nii.image.convert('RGBA')), *pad_box)
                ctx.paint()
                if nii.overlay is not None:
                    ctx = nii.overlay.draw_to_context(ctx, nii.image, cbar, fontsize, linecolor, linewidth, box,
                                                      pad_box, **cbar_kwargs)
            surface.finish()

    def get_image(self, origin=(0, 0, 0), layout='all', height=400, squeeze=False, coord_sys=None, resizing=None,
                  glass_mode=None, cmap=None, transp_if=None, qrange=None, vrange=None, equal_hist=False,
                  is_atlas=False, alpha=.5, crosshair=False, fpath=False, coordinates=False, header=False,
                  histogram=False, cbar=False, title=None, fontsize=20, linecolor='white', linewidth=2, tmp_height=None,
                  nrows=None, as_array=False, **cbar_kwargs):
        origin = len(self) * [origin] if isinstance(origin[0], (int, float, np.integer, np.floating)) else origin
        aspect_ratios = self.get_median_aspect_ratios() if squeeze else None
        title_list = title if isinstance(title, list) else len(self) * [title]
        self.shape = optimal_shape(len(self), layout) if nrows is None else (nrows, int(np.ceil(len(self) / nrows)))
        self.patches = []
        # if tmp_height is None:
        #     print(height)
        nii_tmp_height = None if tmp_height is None else tmp_height // self.shape[0]
        for nii, org, ttl in zip(self.niis, origin, title_list):
            im = nii.get_image(org, layout, height // self.shape[0], aspect_ratios, coord_sys, resizing, glass_mode,
                               cmap, transp_if, qrange, vrange, equal_hist, is_atlas, alpha, crosshair, fpath,
                               coordinates, header, histogram, cbar, ttl, fontsize, linecolor, linewidth,
                               nii_tmp_height, **cbar_kwargs)
            self.patches.append(im)
        self.boxes = get_grid_boxes(sizes=[im.size for im in self.patches], ncols=self.shape[1])
        pad_color = cbar_kwargs.get('cbar_pad_color', 'k')
        im = compose_image(self.patches, self.boxes, pad_color) if len(self) > 1 else self.patches[0]
        is_array = isinstance(im, np.ndarray)
        return Image.fromarray(im) if is_array and not as_array else to_numpy(im) if not is_array and as_array else im

    def get_median_aspect_ratios(self):
        aspect_ratios = [list(nimg.nics[0].aspect_ratios.values()) for nimg in self.niis]
        median_aspect_ratios = np.median(np.array(aspect_ratios), axis=0)
        return {plane: aspect_ratio for plane, aspect_ratio in zip(PLANES, median_aspect_ratios)}


def compose_image(images, boxes, pad_color='k'):
    mode = images[0].mode
    size = (max([box[2] for box in boxes]), max([box[3] for box in boxes]))
    color_value = get_color_values(pad_color, len(mode))
    im = Image.new(mode, size, color_value)
    for box, box_im in zip(boxes, images):
        im.paste(box_im, box)
    return im


def get_grid_boxes(sizes, ncols):
    widths = [size[0] for size in sizes]
    widths = [widths[i:i + ncols] for i in range(0, len(widths), ncols)]
    row_widths = [sum(ws) for ws in widths]
    pads = [(max(row_widths) - sum(ws)) / (len(ws) - 1) if len(ws) > 1 else 0 for ws in widths]
    boxes = []
    for row, (ws, pad) in enumerate(zip(widths, pads)):
        boxes += [(int(sum(ws[:i]) + i * pad), row * sizes[0][1]) for i in range(len(ws))]
    return tuple([(box[0], box[1], box[0] + size[0], box[1] + size[1]) for box, size in zip(boxes, sizes)])


def optimal_shape(n, layout):
    layout_string = LAYOUT_STRINGS[layout] if layout in LAYOUT_STRINGS else layout
    width = len(layout_string.split('|')) - .5 * len([c for c in layout_string if c == '/'])
    nrows_approx = max(1, int((width * n) ** .5))
    ncols = max(1, int(np.ceil(n / nrows_approx)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols
