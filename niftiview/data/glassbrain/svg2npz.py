# import re
# import numpy as np
# from svgpathtools import svg2paths
# VIEWS = ['side', 'front', 'top']
# ZOOM = {'side': (.001636, .002), 'front': (.00205, .001993), 'top': (.001914, .001633)}  # HAND-TUNED TO ALIGN
# TRANS = {'side': (.0387, -.0171), 'front': (1.291, -.0835), 'top': (.0621, .5577)}  # LINES TO DEFAULT TEMPLATE
#
# def bezier_curve(start, control1, control2, end, num_points):
#     points = []
#     for t in range(num_points + 1):
#         t /= num_points
#         x = (1 - t)**3 * start[0] + 3 * (1 - t)**2 * t * control1[0] + 3 * (1 - t) * t**2 * control2[0] + t**3 * end[0]
#         y = (1 - t)**3 * start[1] + 3 * (1 - t)**2 * t * control1[1] + 3 * (1 - t) * t**2 * control2[1] + t**3 * end[1]
#         points.append((x, y))
#     return points
#
#
# def load_svg_paths_as_lines(filepath, n_points=5):
#     paths, headers = svg2paths(filepath)
#     paths = [path.d() for path in paths]
#     lines = []
#     for path, header in zip(paths, headers):
#         style = {s.split(':')[0]: s.split(':')[1] for s in header['style'].split(';')}
#         kwargs = {'width': float(style['stroke-width']), 'fill': style['stroke']}
#         commands = re.findall(r'[MmZzLlHhVvCcSsQqTtAa]', path)
#         coords = re.split(r'[MmZzLlHhVvCcSsQqTtAa]', path)[1:]
#         coords = [list(map(float, re.findall(r'-?\d+\.?\d*', coord))) for coord in coords]
#         points = []
#         for i, (command, coord) in enumerate(zip(commands, coords)):
#             points += bezier_curve(points[-1], coord[:2], coord[2:4], coord[4:], n_points) if command == 'C' else [tuple(coord)]
#             lines.append({'xy': points, **kwargs})
#     return lines
#
#
# def normalize_lines_to_template(lines, zoom, translation):
#     out_lines = []
#     for line in lines:
#         kwargs = {k: v for k, v in line.items() if k != 'xy'}
#         xy = [(zoom[0] * x + translation[0], zoom[1] * y + translation[1]) for x, y in line['xy']]
#         out_lines.append({'xy': xy, **kwargs})
#     return out_lines
#
#
# def svgs_to_npys(directory='.', n_points=5):
#     for view, plane in zip(VIEWS, ('sagittal', 'coronal', 'axial')):
#         lines = load_svg_paths_as_lines(f'{directory}/brain_schematics_{view}.svg', n_points)
#         normalized_lines = normalize_lines_to_template(lines, ZOOM[view], TRANS[view])
#         arguments, points = [], []
#         for line in normalized_lines:
#             color = int(line['fill'][-2:], 16)
#             arguments.append([len(line['xy']), line['width'], color])  # arguments columns: n_points, width, color
#             points += line['xy']  # points columns: x, y
#         arguments = np.array(arguments, dtype=np.float32)
#         points = np.array(points, dtype=np.float32)
#         np.savez_compressed(f'lines_{plane}', arguments=arguments, points=points)
#
#
# if __name__ == '__main__':
#     svgs_to_npys('/home/lfisch/Projects/niftiview/niftiview/data/glassbrain')
