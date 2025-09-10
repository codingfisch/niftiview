![niftiview_logo_transparent_small](https://repository-images.githubusercontent.com/875620357/02caf846-cb7c-402d-a598-39f6ea64745e)

The **easiest** way to **view volumetric images** in **Python** 👩‍💻 **Install** it via `pip install niftiview`!
 
If `pip install niftiview` results in an error, try fixing it via
- `apt install libcairo2-dev pkg-config python3-dev` (on Linux)
- `brew install cairo pkg-config tcl-tk python-tk` (on macOS)

`niftiview` stands behind
- [NiftiView](https://github.com/codingfisch/niftiview-app), the viewer **app** with the cutest desktop icon 🧠 Download it [here](https://github.com/codingfisch/niftiview-app/releases/)!
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
![niftiimage](https://github.com/user-attachments/assets/e31bff70-36b9-4011-a4fc-5512e739e644)

`NiftiImageGrid` can display **multiple images** in a nice **grid layout**
```python
from niftiview import TEMPLATES, NiftiImageGrid

niigrid = NiftiImageGrid([TEMPLATES['ch2'], TEMPLATES['T1']])
# niigrid = NiftiImageGrid(['/path/to/your/nifti1.nii.gz', 
#                           '/path/to/your/nifti2.nii.gz'])
im = niigrid.get_image(layout='sagittal++', nrows=1)
im.show()
```
![niftigrid](https://github.com/user-attachments/assets/99e0f520-a1b0-4cbf-9b34-51a27abb84cd)

Behind the scenes, `niftiview` uses **three main classes** that build on each other
- `NiftiCore`: Puts image slices of the 3D image in a 2D `numpy.ndarray`...
- `NiftiImage`: ...applies a colormap to the array, converts it to a `PIL.Image` and adds overlays...
- `NiftiImageGrid`: ...puts the images in a grid

Study the [**examples**](https://github.com/codingfisch/niftiview/tree/main/examples) 🧑‍🏫 to **fully understand** `niftiview` and create an image like this

![sci_gray](https://github.com/user-attachments/assets/f49bec26-3dcb-43e2-b2e7-a9862006b956)

as a vector graphic─PDF, SVG, EPS or PS format─with high image quality (e.g. for scientific publication 👨‍🔬)

## `niftiview-cli` 🖥️
`pip install niftiview` also installs the **command line utility** that given filepath(s) or a filepattern...
```bash
niftiview-cli -i /path/to/niftis/*.nii.gz -o /path/to/output/folder --gif
```
...saves **PNG**s or **GIF**s. To showcase its versatility, here is a compilation of created GIFs

![niceviews](https://github.com/user-attachments/assets/e2c3648b-1a11-40e9-9230-834a66730f36)

Explore all the available options by running `niftiview-cli --help`

P.S. If you're interested in creating 🧠 tissue maps like the ones shown at the end of the GIF, check out [deepmriprep](https://github.com/wwu-mmll/deepmriprep)!
