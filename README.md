# DistillAnywhere
General framework for generating diverse assets using Score Distillation.

## How to setup

``` bash
git clone https://github.com/32V/DistillAnywhere.git --recursive
cd DistillAnywhere
conda create -n distillanywhere python=3.9 -y
conda activate distillanywhere
pip install -r requirements.txt
pip install third_party/gsplat/
pip install third_party/nvdiffrast
```

(It may not work since I've never tested it)

## How to run

```bash
python main.py --config config/image/image.yaml text_prompt="A DSLR photo of a cat"
python main.py --config config/image/latent_image_sdi.yaml text_prompt="A DSLR photo of a cat"
python main.py --config config/gs/gs_mv_2000.yaml text_prompt="A DSLR photo of a rabbit on a pancake"
python main.py --config config/gs/gs_mv_2000.yaml text_prompt="a DSLR photo of face of a man, best quality, high quality, extremely detailed, good geometry" mesh_path=face.obj
```
