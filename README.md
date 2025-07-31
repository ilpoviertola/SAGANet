# SAGANet: Video Object Segmentation-Aware Audio Generation

[Ilpo Viertola](https://scholar.google.com/citations?user=gGWNg4EAAAAJ&hl=en), [Vladimir Iashin](https://scholar.google.com/citations?user=rh8_sSkAAAAJ&hl=en), and [Esa Rahtu](https://scholar.google.com/citations?user=SmGZwHYAAAAJ&hl=en)

[[Project Page](saganet.notion.site)]

DAGM German Conference on Pattern Recognition (GCPR) 2025.

Thanks for the team of [MMAudio](https://github.com/hkchengrex/MMAudio). Our work is based on their codebase, licensed under the MIT License.

## Cite

```bibtex
@inproceedings{poviertola2025saganet,
  title={SAGANet: Video Object Segmentation-Aware Audio Generation},
  author={Viertola, Ilpo and Iashin, Vladimir and Rahtu, Esa},
  booktitle={DAGM German Conference on Pattern Recognition (GCPR)},
  year={2025}
}
```

## Installation

Tested with Ubuntu 22.04.5.

### 1. Install following dependencies to your virtual environment

- Python 3.9
- PyTorch 2.6

### 2. Clone the repository

```bash
git clone git@github.com:ilpoviertola/SAGANet.git
```

### 3. Install the requirements

```bash
cd SAGANet
pip install -r requirements.txt
```

All the necessat weights are downloaded automatically when you run the demo.

## Demo

Running the demo takes around 4.5 GB of GPU memory. Below is an example command. Explore the `demo.py` file for more options.

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 demo.py --variant saganet_small --video ./example/Vid_09_Jesus_tpt_vn_159092_164092.mp4 --mask_video ./example/Masks_02_vn_09_Jesus_159092_164092.mp4 --prompt violin --negative_prompt trumpet
```

## Segmented Music Solos (SMS) Dataset

We propose a new dataset to facilitate training of video segmentation-aware audio models. Check out the [project page](https://saganet.notion.site/Segmented-Music-Solos-SMS-23e82e9d170080369505c222bda09447) for up-to-date information how to obtain the dataset.

## TODO

- [ ] Add on-the-fly segmentation mask generation from user prompts.
