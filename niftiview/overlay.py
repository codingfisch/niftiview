import cairo
import numpy as np
from pathlib import Path
from colorbar.cbar import VERTICAL, PAD, PAD_COLOR, X, Y, add_padding, CBar
from colorbar.utils import draw, to_numpy, draw_objects, get_source_rgba, CMap

from .core import PLANES
from .utils import DATA_PATH, get_filestem
FONTS = {get_filestem(fp): str(fp) for fp in sorted(Path(f'{DATA_PATH}/fonts').glob('*.ttf'))}
FONT_DEFAULT = 'DejaVuSans'


class Overlay:
    def __init__(self, nics, cmap=None, vmin=0, vmax=1, font=None, extra_lines=None, extra_texts=None):
        self.nics = nics
        self.cbar = CBar(CMap(cmap) if isinstance(cmap, str) else cmap, vmin, vmax)  # use CBar(cmap, vmin, vmax) for CBar with transparency?
        self.font = font or FONT_DEFAULT
        self.extra_lines = [] if extra_lines is None else extra_lines
        self.extra_texts = [] if extra_texts is None else extra_texts
        self._lines = []
        self._texts = []

    def save(self, filepath, image, crosshair=False, fpath=False, coordinates=False, header=False, histogram=False,
             cbar=False, title=None, fontsize=30, linecolor='white', linewidth=1, **cbar_kwargs):
        size = add_padding(image.copy(), **{k.replace('cbar_', ''): v for k, v in cbar_kwargs.items()}).size
        self.set_draw_objects(crosshair, fpath, coordinates, header, histogram, title, linewidth, fontsize, size)
        SurfaceClass = get_surface_class(filepath.split('.')[-1])
        with SurfaceClass(filepath, size[0], size[1]) as surface:
            ctx = cairo.Context(surface)
            ctx = draw_background(ctx, **cbar_kwargs)
            pad_box = get_pad_box(**cbar_kwargs)
            ctx.set_source_surface(from_pil(image.convert('RGBA')), *pad_box)
            ctx.paint()
            self.draw_to_context(ctx, image, cbar, fontsize, linecolor, linewidth, pad_box=pad_box, **cbar_kwargs)
            surface.finish()

    def draw_to_context(self, ctx, im, cbar, fontsize=30, linecolor='white', linewidth=1, box=(0, 0), pad_box=(0, 0),
                        **cbar_kwargs):
        draw_objects(ctx, lines=self._lines, texts=self._texts, linecolor=linecolor, box=pad_box)
        if cbar:
            kwargs = {k.replace('cbar_', ''): v for k, v in cbar_kwargs.items()}
            self.cbar.draw(to_numpy(im), apply_cmap=False, fontsize=fontsize, linecolor=linecolor,
                           linewidth=linewidth, asarray=False, **kwargs)
            lw = max(linewidth, 1.3)
            xy = self.cbar._rectangle['xy']
            patches = [{'xy': [xy[0] + lw, xy[1] + lw],
                        'im': self.cbar._bar.crop((lw + 1, lw + 1, self.cbar._bar.size[0], self.cbar._bar.size[1]))}]
            draw_objects(ctx, patches, [self.cbar._rectangle], self.cbar._lines, self.cbar._texts,
                         linecolor=linecolor, box=box)
        return ctx

    def draw(self, im, crosshair=False, fpath=False, coordinates=False, header=False, histogram=False, cbar=False,
             title=None, fontsize=30, linecolor='white', linewidth=1, **cbar_kwargs):
        self.set_draw_objects(crosshair, fpath, coordinates, header, histogram, title, linewidth, fontsize, im.size)
        im = draw(im, lines=self._lines, texts=self._texts, linecolor=linecolor)
        if cbar:
            kwargs = {k.replace('cbar_', ''): v for k, v in cbar_kwargs.items()}
            im = self.cbar.draw(to_numpy(im), apply_cmap=False, fontsize=fontsize, linecolor=linecolor,
                                linewidth=linewidth, asarray=False, **kwargs)
        return im

    def set_draw_objects(self, crosshair, fpath, coordinates, header, histogram, title, linewidth, fontsize, im_size):
        margin = fontsize // 5
        self._texts, self._lines = [], []
        self._texts += self.get_filepath_texts(fpath, margin, fontsize) if fpath else []
        self._texts += self.get_coordinates_texts(margin, fontsize) if coordinates else []
        self._texts += self.get_header_text(margin, fpath > 0, fontsize) if header else []
        self._texts += self.get_title_text(title, margin, im_size[0], int(1.5 * fontsize)) if title is not None else []
        if histogram:
            texts, lines = self.get_histogram(margin, int(fpath > 0) + 4 * header, fontsize)
            self._texts += texts
            self._lines += lines
        if crosshair:
            multiline = len(set(self.nics[0].image_planes)) != len(self.nics[0].image_planes)
            self._lines += self.get_multi_lines(linewidth) if multiline else self.get_cross_lines(linewidth)
        self._lines += self.extra_lines
        self._texts += self.extra_texts

    def get_title_text(self, text, margin, width, fontsize, anchor='ma'):
        return [{'xy': (width // 2, margin), 'text': text, 'fontsize': fontsize, 'anchor': anchor}]

    def get_filepath_texts(self, fpath, margin, fontsize):
        filepath = '' if self.nics[-1].filepath is None else self.nics[-1].filepath
        text = filepath if isinstance(fpath, bool) or filepath == '' else '/'.join(Path(filepath[1:]).parts[-fpath:])
        return [{'xy': (margin, margin), 'text': text, 'fontsize': fontsize}]

    def get_coordinates_texts(self, margin, fontsize, anchor='ld'):
        texts = []
        for kw in self.nics[0]._image_props:
            xy = [kw['box'][0] + margin, kw['box'][1] + kw['size'][1] - margin]
            origin = self.nics[0].affine @ np.append(kw['idx'][:3], 1)
            dim = PLANES.index(kw['plane'])
            texts.append({'xy': xy, 'text': f'{"xyz"[dim]} = {int(origin[dim])}', 'fontsize': fontsize, 'anchor': anchor})
        return texts

    def get_header_text(self, margin, row, fontsize):
        x = row * fontsize
        shape_str = ' x '.join([str(s) for s in self.nics[-1].shape])
        text = [{'xy': (margin, margin + x), 'text': f'Array: {shape_str} x {self.nics[-1].dtype}', 'fontsize': fontsize}]#, 'anchor': anchor}]
        for i in range(3):
            text_row = 'Affine: (' if i == 0 else '('
            text_cols = []
            for i_col, v in enumerate(self.nics[-1].affine[i]):
                text_cols.append(f'{v:.2f}' if i_col < 3 else f'{v:.4f}'[:6])
            text_row += ', '.join(text_cols) + ')'
            text.append({'xy': (4 * fontsize if i > 0 else margin, margin + x + (i + 1) * fontsize), 'text': text_row,
                         'fontsize': fontsize})
        return text

    def get_histogram(self, margin, row, fontsize, colon_text='Histogram: ', step=4):
        y = fontsize * row
        percentiles = self.nics[-1].quantile(np.linspace(0, 1, 10001))
        min_text, max_text = f'{percentiles[0]:.5g}', f'{percentiles[-1]:.5g}'
        max_x = fontsize * 15
        texts = [{'xy': [margin, y + margin], 'text': colon_text, 'fontsize': fontsize},
                 {'xy': [margin + len(colon_text) * fontsize // 2, y + margin], 'text': min_text, 'fontsize': fontsize},
                 {'xy': [3 * margin + max_x, y + margin], 'text': max_text, 'fontsize': fontsize}]
        offset = margin + fontsize * len(colon_text + min_text) // 2
        size = max_x - offset
        old_quantile, lines = 0, []
        for i in range(step, size + step, 4):
            v = (i / size) * (percentiles[-1] - percentiles[0]) + percentiles[0]
            quantile = (v > percentiles).mean()
            quantile_diff = quantile - old_quantile
            fill = min(int(255 * quantile_diff * size / 4), 255)
            fill = fill if self.cbar.cmap.name == 'matlab:gray' and not self.cbar.cmap.force_rgba else tuple(3 * [fill] + [255])
            lines.append({'xy': [offset + margin + i, y + 2 * margin, offset + margin + i, y + margin + fontsize],
                          'width': 2, 'fill': fill})
            old_quantile = quantile
        return texts, lines

    def get_cross_lines(self, width):
        lines = []
        for kw in self.nics[0]._image_props:
            dims = [dim for dim, p in enumerate(PLANES) if p != kw['plane']]
            rel_pos = (np.array(kw['idx'])[:3] / self.nics[0].shape[:3])[dims]
            box, size = kw['box'], kw['size']
            abs_pos = (int(round(rel_pos[0] * size[0])), int(round((1 - rel_pos[1]) * size[1])))
            lines.append({'xy': (box[0] + abs_pos[0], box[1], box[0] + abs_pos[0], box[1] + size[1]), 'width': width})
            lines.append({'xy': (box[0], box[1] + abs_pos[1], box[0] + size[0], box[1] + abs_pos[1]), 'width': width})
        return lines

    def get_multi_lines(self, linewidth):
        planes = self.nics[0].image_planes
        idxs = self.nics[0].image_indices
        lines = []
        if planes[0] != planes[-1]:
            vertical = planes[0][0] != planes[-1][0] in ['sc', 'sa', 'cs']
            box = self.nics[0].image_boxes[-1]
            size = self.nics[0].image_sizes[-1]
            for plane, idxs in zip(planes[:-1], idxs[:-1]):
                dim = PLANES.index(plane)
                rel_pos = idxs[dim] / self.nics[0].shape[dim]
                abs_pos = rel_pos * size[0] if vertical else (1 - rel_pos) * size[1]
                if vertical:
                    line = (box[0] + abs_pos, box[1], box[0] + abs_pos, box[1] + size[1])
                else:
                    line = (box[0], box[1] + abs_pos, box[0] + size[0], box[1] + abs_pos)
                lines.append({'xy': line, 'width': linewidth})
        return lines


def get_surface_class(filetype='pdf'):
    return {'svg': cairo.SVGSurface, 'eps': cairo.PSSurface, 'ps': cairo.PSSurface, 'pdf': cairo.PDFSurface}[filetype]


def draw_background(ctx, cbar_pad_color=PAD_COLOR, **kwargs):
    ctx.set_source_rgba(*get_source_rgba(cbar_pad_color))
    ctx.paint()
    return ctx


def get_pad_box(cbar_vertical=VERTICAL, cbar_pad=PAD, cbar_x=X, cbar_y=Y, **kwargs):
    return cbar_pad if cbar_vertical and cbar_x < .5 else 0, cbar_pad if not cbar_vertical and cbar_y < .5 else 0


def from_pil(im):
    arr = bytearray((im if im.mode == 'RGBA' else im.convert('RGBA')).tobytes('raw', 'BGRa'))
    return cairo.ImageSurface.create_for_data(arr, cairo.FORMAT_ARGB32, im.width, im.height)
