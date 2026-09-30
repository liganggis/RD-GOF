# RD-GOF：Rasterization-Driven Gaussian Opacity Field for Efficient High-Fidelity Surface Reconstruction

![teaser](assets/synthetic0.png)

This project aim to introduce a rasterization-driven method for constructing Gaussian opacity fields, thereby achieving high-fidelity and efficient surface reconstruction from 3D Gaussian Splatting (3DGS). We will release all source code upon paper acceptance.

## Dataset

### Download Datasets

We use the preprocessed DTU dataset from [2DGS](https://github.com/hbb1/2d-gaussian-splatting?tab=readme-ov-file#quick-examples), the official Tanks and Temples dataset, and the official Mip-NeRF 360 dataset. Here are the instructions for each.

- [DTU](https://drive.google.com/file/d/1ODiOu72tAGPTnhVn0cFZ9MvymDgcoHxQ/view?usp=drive_link) dataset (2DGS pre-processed)
  - To get the ground-truths, you need also to download the [Points.zip](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) and [SampleSet.zip](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip).
- [Tanks and Temples](https://www.tanksandtemples.org/download/) dataset (Official)
  - Ground truth, image set, camera poses, alignment, and cropfiles are required.
  - Following Neuralangelo [instructions](https://github.com/NVlabs/neuralangelo/blob/main/DATA_PROCESSING.md#tanks-and-temples-dataset) while removing [L230](https://github.com/NVlabs/neuralangelo/blob/94390b64683c067c620d9e075224ccfe582647d0/projects/neuralangelo/scripts/convert_tnt_to_json.py#L230) to process the scenes with COLMAP.
  - Due to substantial inaccurate estimation existing, we recommend using the 2DGS pre-processed `Courthouse` scene from [here](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main) as a start to slightly relieve this problem.
- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset (Official)

And the data structure should be organized as follows:

```shell
data/
├── 360_v2
│   ├── bicycle/
│   ├── bonsai/
│   ├── ...
├── DTU
│   ├── scan24
│   ├── ...
├── TNT
│   ├── Barn
│   ├── Caterpillar
│   ├── Courthouse
│   ├── Ignatius
│   ├── Meetingroom
│   ├── Truck
```


## Pipeline

![pipeline](assets/mesh-main.png)


## Run

### Environment

```shell
git clone https://github.com/liganggis/RD-GOF
cd RD-GOF

conda create -n RD-GOF-env python=3.10
conda activate RD-GOF-env

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

# install dependencies
pip install -r requirements.txt

# install submodules
Unzip the zip file in the submodules folder, then
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# install Marching Tetrahedra
conda install -y conda-forge::cgal
pip install submodules/tetra_triangulation
```
### Train

We have provided the script in the folder (https://github.com/liganggis/RD-GOF/script/) that were used to generate the table in the paper.

## Acknowledgments

...


## BibTex


And thanks to the authors of [3D Gaussians](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [PGSR](https://github.com/zju3dv/PGSR) and[GGGS](https://baowenz.github.io/geometry_grounded_gaussian_splatting/) for their excellent code, please consider citing these repositories.
