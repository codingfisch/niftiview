import argparse
import numpy as np
from tqdm import tqdm
from colorbar.utils import to_numpy

from niftiview.grid import NiftiImageGrid
from niftiview.core import PLANES, PLANE_DICT
from niftiview.utils import get_filestem


def save_gif(nii, filepath, origin=None, layout='sagittal++', duration=20, loop=0, start=None, stop=None, **kwargs):
    frames = get_image_frames(nii, origin, layout, start, stop, **kwargs)
    if filepath.endswith('.gif'):
        frames[0].save(filepath, append_images=frames[1:], save_all=True, duration=duration, loop=loop)
    elif filepath.endswith('.png'):
        for i, frame in enumerate(frames):
            frame.save(f'{filepath[:-4]}_{i}.png')


def get_image_frames(nii, origin=None, layout='sagittal++', start=None, stop=None, **kwargs):
    is_grid = isinstance(nii, NiftiImageGrid)
    org = np.zeros(4) if origin is None else origin
    org = np.stack(len(nii) * [org])
    shape = nii.niis[0].nics[0].array.shape if is_grid else nii.nics[0].array.shape
    dim = 3 if shape[3] > 1 else PLANES.index(PLANE_DICT[layout[0]])
    if start is None:
        if shape[3] > 1:
            start = 0
        elif is_grid:
            start = int(min([nii.nics[0].get_origin_bounds()[0, dim] for nii in nii.niis]))
        else:
            start = int(nii.nics[0].get_origin_bounds()[0, dim])
    if stop is None:
        if shape[3] > 1:
            stop = shape[3]
        elif is_grid:
            stop = int(max([nii.nics[0].get_origin_bounds()[1, dim] for nii in nii.niis]))
        else:
            stop = int(nii.nics[0].get_origin_bounds()[1, dim])
    org[:, dim] = start
    frame = nii.get_image(org if is_grid else org[0], layout, **kwargs).copy()
    frames = []
    for i in range(start + 1, stop, 1):
        org[:, dim] = i
        new_frame = nii.get_image((org if is_grid else org[0]).tolist(), layout, **kwargs).copy()
        if not np.array_equal(to_numpy(new_frame), to_numpy(frame)):
            frames.append(new_frame)
        frame = new_frame.copy()
    return frames


def save_images_or_gifs(in_filepaths, out_dir, gif=True, max_samples=9, origin=None, layout='sagittal++', duration=20,
                        loop=0, start=None, stop=None, **kwargs):
    origin = origin or [0, 0, 0, 0]
    in_filepaths = [[fp] for fp in in_filepaths] if isinstance(in_filepaths[0], str) else in_filepaths
    for i in tqdm(range(0, len(in_filepaths), max_samples)):
        filepaths = in_filepaths[i:i + max_samples]
        niigrid = NiftiImageGrid(filepaths)
        filestem = get_filestem(filepaths[0][0])
        if len(filepaths) > 1:
            filestem += '_' + get_filestem(filepaths[-1][0])
        if gif:
            save_gif(niigrid, f'{out_dir}/{filestem}.gif', origin, layout, duration, loop, start, stop, **kwargs)
        else:
            niigrid.get_image(origin, layout, **kwargs).save(f'{out_dir}/{filestem}.png')


def main():
    bool_opt = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input filepath or pattern', required=True, nargs='+')
    parser.add_argument('-o', '--output', help='Output folder', required=False, type=str, default=None)
    parser.add_argument('-g', '--gif', help='If this flag is set save GIFs otherwise PNGs', action=bool_opt)
    parser.add_argument('-m', '--max_samples', help='Max. samples per GIF/Image', type=int, default=4)
    parser.add_argument('-l', '--layout', help='Layout', type=str, default='sagittal++')
    parser.add_argument('--height', help='Image height (in pixels)', type=int, default=600)
    parser.add_argument('--origin', help='Origin', nargs='+', type=int, default=None)
    parser.add_argument('--coord_sys', help='Coordinate system', type=str, default='array_idx')
    parser.add_argument('--squeeze', help='Squeeze images into regular grid', type=bool, default=False, action=bool_opt)
    parser.add_argument('--resizing', help='Interpolation mode', type=str, default=1)
    parser.add_argument('--cmap', help='Colormap for image', type=str, default='gray')
    parser.add_argument('--transp_if', help='Transparency condition for image', type=str, default=None)
    parser.add_argument('--vrange', help='Value range', nargs=2, type=float, default=None)
    parser.add_argument('--qrange', help='Value range (based on quantiles)', nargs=2, type=float, default=(0.01, 0.99))
    parser.add_argument('--equal_hist', help='Equal histogram', action=bool_opt, default=False)
    parser.add_argument('--is_atlas', help='Is atlas', action=bool_opt, default=False)
    parser.add_argument('--alpha', help='Alpha value', type=float, default=0.5)
    parser.add_argument('--crosshair', help='Show crosshair', action=bool_opt, default=False)
    parser.add_argument('--fpath', help='Show filepath', type=int, default=0)
    parser.add_argument('--coordinates', help='Show coordinates', action=bool_opt, default=False)
    parser.add_argument('--header', help='Show header', action=bool_opt, default=False)
    parser.add_argument('--histogram', help='Show histogram', action=bool_opt, default=False)
    parser.add_argument('--cbar', help='Show colorbar', action=bool_opt, default=False)
    parser.add_argument('--title', help='Show title', type=str, nargs='+', default=None)
    parser.add_argument('--fontsize', help='Font size', type=int, default=20)
    parser.add_argument('--linecolor', help='Line color', type=str, default='white')
    parser.add_argument('--linewidth', help='Line width', type=int, default=2)
    parser.add_argument('--nrows', help='Number of rows', type=int, default=None)
    parser.add_argument('--cbar_vertical', help='Vertical colorbar', action=bool_opt, default=True)
    parser.add_argument('--cbar_pad', help='Colorbar padding', type=float, default=0)
    parser.add_argument('--cbar_pad_color', help='Colorbar padding color', type=str, default='black')
    parser.add_argument('--cbar_x', help='Colorbar x position', type=float, default=0.9)
    parser.add_argument('--cbar_y', help='Colorbar y position', type=float, default=0.5)
    parser.add_argument('--cbar_length', help='Colorbar length', type=float, default=0.8)
    parser.add_argument('--cbar_width', help='Colorbar width', type=float, default=0.05)
    parser.add_argument('--cbar_label', help='Colorbar label', type=str, default=None)
    parser.add_argument('--cbar_ticks', help='Colorbar ticks', type=float, nargs='+', default=None)
    args = parser.parse_args()

    in_fpaths, out_dir, title = args.input, args.output, args.title
    delattr(args, 'input')
    delattr(args, 'output')
    delattr(args, 'title')
    print(args)
    title = title[0] if len(title) else title
    save_images_or_gifs(in_filepaths=in_fpaths, out_dir=out_dir, title=title, **vars(args))


if __name__ == '__main__':
    main()
