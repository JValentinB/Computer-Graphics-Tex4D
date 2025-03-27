# Computer-Graphics-Tex4D

Based on [Tex4D](https://github.com/ZqlwMatt/Tex4D).

We are developing a Tex4d-Addon for Blender by replicating the model described by [Bao et al](https://arxiv.org/pdf/2410.10821)

## Installation

### Tex4D

Using requirements.txt:

```bash
git clone https://github.com/JValentinB/Computer-Graphics-Tex4D.git
cd Computer-Graphics-Tex4D
conda create -n tex4d python=3.8
conda activate tex4d
pip install -r requirements.txt
```
Then install PyTorch3D through the following URL (check and replace your CUDA & PyTorch version)
```bash
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt201/download.html
```

Alternatively, use conda directly:

```bash
git clone https://github.com/JValentinB/Computer-Graphics-Tex4D.git
cd Computer-Graphics-Tex4D
conda env create -f environment.yaml
conda activate tex4d
```
### Blender Add-on

- Download BlenderAddon.zip
- Blender>Edit>Preferences>Add-ons> Install via zip

## Running the Tex4D Server

```bash
conda activate tex4d
python app.py
```