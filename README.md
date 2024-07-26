# DistillAnywhere
General framework for generating diverse assets using Score Distillation.

## How to setup

``` bash
git clone https://github.com/32V/DistillAnywhere.git --recursive
cd DistillAnywhere
conda create -n distillanywhere python=3.9 -y
conda activate distillanywhere
pip install -r requirements.txt -y
pip install third_party/gsplat/
pip install third_party/nvdiffrast
pip install third_party/mvdream_diffusers
```

(It may not work since I've never tested it)

## How to run

```bash
python main.py --config config/image/latent_image_sdi.yaml text_prompt="A DSLR photo of a cat"
python main.py --config config/gs/gs_mv_2000.yaml text_prompt="A DSLR photo of a rabbit on a pancake"
```
