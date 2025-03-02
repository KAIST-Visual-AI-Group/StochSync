# StochSync: Stochastic Diffusion Synchronization for Image Generation in Arbitrary Spaces 

<p align="center">
    <a class="active text-decoration-none" href="">Kyeongmin Yeo</a><sup> *</sup>,  &nbsp;
    <a class="active text-decoration-none" href="https://jh27kim.github.io/">Jaihoon Kim</a><sup> *</sup>, &nbsp;
    <a class="active text-decoration-none" href="https://mhsung.github.io/">Minhyuk Sung</a> &nbsp;
</p>
<p align="center">
  <span class="author-block">KAIST</span>&nbsp;
</p>

<p align="center">
  <span class="author-block"><sup>*</sup>Equal contribution</span>&nbsp;
</p>

---

<p align="center">
  <a href="https://arxiv.org/abs/2501.15445">
    <img src="https://img.shields.io/badge/arXiv-2501.15445-b31b1b.svg?logo=arXiv">
  </a>&nbsp;
  <a href="https://arxiv.org/pdf/2501.15445">
    <img src="https://img.shields.io/badge/paper-b31b1b.svg?logo=arXiv&color=6c68d4">
  </a>&nbsp;
  <a href="https://stochsync.github.io/">
    <img src="https://img.shields.io/badge/project page-blue?logo=github">
  </a>
</p>

![Teaser Image](assets/teaser.png)

---

## Introduction

We propose $\texttt{StochSync}$, a method for generating images in arbitrary spaces—such as 360° panoramas or textures on 3D surfaces—using a **pretrained image diffusion model**. The main challenge is bridging the gap between the 2D images understood by the diffusion model (instance space $\mathcal{X}$) and the target space for image generation (canonical space $\mathcal{Z}$). Unlike previous methods that struggle without strong conditioning or lack fine details, $\texttt{StochSync}$ combines the strengths of Diffusion Synchronization and Score Distillation Sampling to perform effectively even with weak conditioning. Our experiments show that $\texttt{StochSync}$ outperforms prior finetuning-based methods, especially in 360° panorama generation.

---

## Environment and Requirements

### Tested Environment
- **Python:** 3.9
- **CUDA:** CUDA 12.1
- **GPU:** Tested on NVIDIA GTX3090Ti and A6000

### Installation Steps

1. **Clone the Repository with Submodules:**

   ```bash
   git clone --recursive https://github.com/KAIST-Visual-AI-Group/StochSync.git & cd StochSync
   ```

2. **Create Conda Environment:**
    ```bash
    conda create -n distillanywhere python=3.9 -y
    conda activate distillanywhere
    ```

3. **Install Core Dependencies:**
    
    First, install PyTorch and xformers compatible with your CUDA environment. For example:
    
    ```bash
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers --force-reinstall --index-url https://download.pytorch.org/whl/cu121
    ```
    
4. **Install Python Dependencies:**
    
    Install remaining dependencies using `requirements.txt` and additional modules in `third_party/`:
    
    ```bash
    pip install -r requirements.txt
    pip install third_party/gsplat/
    pip install third_party/nvdiffrast/
    ```

---

## Usage and Examples

### Running StochSync

We provide several example configurations in the `config/` directory. Below are examples for different applications:

- **Format**:

    ```bash
    python main.py --config "your_config.yaml" root_dir="root_dir_for_results" tag="run_name" text_prompt="your text prompt here"
    ```

- **360° Panorama Generation:**
    
    ```bash
    python main.py --config config/stochsync_panorama.yaml text_prompt="A vibrant urban alleyway filled with colorful graffiti, and stylized lettering on wall"
    ```
    
- **3D Mesh Texturing:**
    
    ```bash
    python main.py --config config/stochsync_mesh.yaml mesh_path="./data/mesh/face.obj" text_prompt="Kratos bust, God of War, god of power, hyper-realistic and extremely detailed."
    ```
    
- **Sphere & Torus Texture Generation:**
    
    ```bash
    python main.py --config config/stochsync_sphere.yaml text_prompt="Paint splatter texture."
    ```

    ```bash
    python main.py --config config/stochsync_torus.yaml text_prompt="Paint splatter texture."
    ```
    

<!-- ### Example Visual Results

Below are placeholders for example results from various applications:

- **Panorama Result:**
- **Mesh Texturing Result:**
- **Sphere/Torus Texture Generation:** -->

---

## Unit Testing

We provide comprehensive unit tests to validate the functionality of our modules. To run the unit tests, execute:

```bash
python run_unit_test.py --extensive --devices {list of gpu indices to use}
```

Test results will be stored in the directory: `unit_test_results/{application}`.

---

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{yeo2025stochsync,
  title={StochSync: Stochastic Diffusion Synchronization for Image Generation in Arbitrary Spaces},
  author={Yeo, Kyeongmin and Kim, Jaihoon and Sung, Minhyuk},
  journal={arXiv e-prints},
  pages={arXiv--2501},
  year={2025}
}

```

---

## Acknowledgements

This repository builds upon several outstanding projects and libraries. We would like to express our gratitude to the developers and contributors of:

- **NVDiffrast**
- **paint-it**
- **gsplat**
- **mvdream**

Their work has been instrumental in the development of StochSync.