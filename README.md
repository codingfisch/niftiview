![niftiview_logo_transparent_small](https://github.com/user-attachments/assets/a03ab906-59cb-4ad6-b526-9774b36bf8c9)

The **easiest** way to **view volumetric images** in **Python** 👩‍💻 **Install** it via `pip install niftiview`! 

`niftiview` stands behind
- [NiftiView](https://github.com/codingfisch/niftiview_app), the viewer **app** with the cutest desktop icon 🧠 Download it [here](https://github.com/codingfisch/niftiview_app)!
- [NiftiWidget](https://github.com/codingfisch/niftiwidget), a **widget** for interactive viewing in [Jupyter](https://jupyter.org/) 👩‍💻🧠 Install it via `pip install niftiwidget`! 

## Usage 💡
**Single images** can be shown via `NiftiImage` 
```python
from niftiview import TEMPLATES, NiftiImage

nii = NiftiImage(TEMPLATES['ch2'])
# nii = NiftiImage('/path/to/your/nifti.nii.gz')
im = nii.get_image()
im.show()
```
`NiftiImageGrid` can display **multiple images** in a nice **grid layout**
```python
from niftiview import TEMPLATES, NiftiImageGrid

niigrid = NiftiImageGrid([TEMPLATES['ch2'], TEMPLATES['T1']])
# niigrid = NiftiImageGrid(['/path/to/your/nifti1.nii.gz', 
#                           '/path/to/your/nifti2.nii.gz'])
im = niigrid.get_image()
im.show()
```
Behind the scenes, `niftiview` uses **three main classes** that build on each other
- `NiftiCore`: Puts image slices of the 3D image in a 2D `numpy.ndarray`...
- `NiftiImage`: ...applies a colormap to the array, converts it to a `PIL.Image` and adds overlays...
- `NiftiImageGrid`: ...puts the images in a grid

To **fully understand** how to use `niftiview`, study the **example notebooks** 🧑‍🏫
- `examples/0_core.ipynb` explaining `NiftiCore`
- `examples/1_image.ipynb` explaining `NiftiImage`
- `examples/2_grid.ipynb` explaining `NiftiImageGrid`

## `niftiview-cli` 🖥️
`pip install niftiview` also installs the **command line utility** that given filepath(s) or a filepattern...
```bash
niftiview-cli -i /path/to/niftis/*.nii.gz -o /path/to/output/folder --gif
```
...saves **PNG**s or **GIF**s. Take a look at all its possible options via `niftiview-cli --help`!
