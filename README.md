# Run Instructions

## What to bring in
* 3D model with geometry + material info
  * .obj + .mtl + texture files
  * .glb
  * .dae + textures
* Python 3 with numpy
* Blender 4.x for rendering

## Run the code
* Sample mesh to point cloud
  * `python code/src/sample.py data/test.obj -o output/points.npy --ply output/points.ply -n 10000 --seed 0`
* Convert points to Gaussian splat PLY
  * `python code/src/convert.py output/points.npy -o output/splat.ply --scale 0.005 --opacity 0.95`
* View the sampled point cloud
  * `python code/src/view_points.py`
  * To view another file: `python code/src/view_points.py path/to/points.npy`
* Render reference images with Blender
  * `blender --background --python code/blender/setup_scene.py -- --mesh data/test.obj --hdri data/hdri.exr --outdir output/renders --cameras output/cameras.json --num-cams 8`

## Outputs
* `output/points.npy`, `output/points.ply`
* `output/splat.ply`
* `output/renders/*.png`, `output/cameras.json`
