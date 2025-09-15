# tracking seabirds

An example notebook demoing how to transform 2D keypoints to a new coordinate system using `movement`.

The data is a short clip of tracked seabirds flying around a boat. The notebook shows how to express the data in a coordinate system aligned with the boat.

## Usage

Create a conda environment and install the latest version of `movement`:

```sh
conda create -n movement-env -c conda-forge movement
```

Activate the environment:
```sh
conda activate movement-env
```

Install additional dependencies for Jupyter Notebooks:

```sh
pip install jupyter jupyterlab ipympl
```

If you wish to use the `movement`GUI, which additionally requires [napari](napari:),
you should replace the first command with:
```sh
conda create -n movement-env -c conda-forge movement napari pyqt
```
You may exchange `pyqt` for `pyside2` if you prefer.
See [napari's installation guide](napari:tutorials/fundamentals/installation.html)
for more information on the backends.


Please refer to the [movement installation guide](https://movement.neuroinformatics.dev/user_guide/installation.html) for more details.

To open the notebook in Jupyter Lab, run from the root of this repository:

```bash
jupyter lab notebook_seabirds.ipynb
```

Alternatively, you can run the notebook in VS Code using the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).


## Next steps
- Scale distances using boat width
- Define ROIs