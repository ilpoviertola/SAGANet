# SAGANet: Video Object Segmentation-Aware Audio Generation

[Ilpo Viertola](https://scholar.google.com/citations?user=gGWNg4EAAAAJ&hl=en), [Vladimir Iashin](https://scholar.google.com/citations?user=rh8_sSkAAAAJ&hl=en), and [Esa Rahtu](https://scholar.google.com/citations?user=SmGZwHYAAAAJ&hl=en)

[[Project Page](https://saganet.notion.site/)][[ArXiv](https://arxiv.org/abs/2509.26604v1)]

DAGM German Conference on Pattern Recognition (GCPR) 2025.

Thanks for the team of [MMAudio](https://github.com/hkchengrex/MMAudio). Our work is based on their codebase, licensed under the MIT License.

## Cite

```bibtex
@inproceedings{ilpoviertola2025saganet,
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

We propose a new dataset to facilitate training of video segmentation-aware audio models.

### 1. Download raw data

Follow the isntructions of a specific dataset to download the raw data. The dataset is a combined and re-processed version of the following datasets:

1. MUSIC21 (https://github.com/roudimit/MUSIC_dataset)
2. AVSSBench (https://github.com/OpenNLPLab/AVSBench)
3. Solos (https://github.com/JuanFMontesinos/Solos)
4. URMP (https://labsites.rochester.edu/air/projects/URMP.html)

### 2. Download metadata and masks

```bash
python training/load_sms.py --solos /path/to/solos --music21 /path/to/music21 --avssbench /path/to/avs-semantic-v2-25fps --urmp /path/to/URMP
```

### 3. Prepare the dataset

Next we need to prepare the dataset for training. First, let's create the focal prompts. In the script, uncomment the `data_cfg` you want to use.

```bash
python training/extract_focal_prompts.py
```

Then, we can extract the audio and video features. Uncomment the `data_cfg` you want to use in the script.

```bash
python training/extract_video_training_latents.py
```

## Training

### Prerequisites

1. Install [av-benchmark](https://github.com/hkchengrex/av-benchmark). We use this library to automatically evaluate on the validation set during training, and on the test set after training.
2. Extract features for evaluation using [av-benchmark](https://github.com/hkchengrex/av-benchmark) for the validation and test set as a [validation cache](https://github.com/ilpoviertola/SAGANet/blob/main/config/data/sms_slurm.yaml#L13) and a [test cache](https://github.com/ilpoviertola/SAGANet/blob/main/config/data/urmp_slurm.yaml#7).
3. You will need ffmpeg to extract frames from videos. Note that `torchaudio` imposes a maximum version limit (`ffmpeg<7`). You can install it as follows:

```bash
conda install -c conda-forge 'ffmpeg<7'
```

### Training command

First, specify the data path(s) in `config/data/base.yaml`. For full training on the base model with two GPUs, use the following command:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 train.py
```

## Evaluation

To evaluate the model on a dataset, use the `batch_eval.py` script. It is significantly more efficient in large-scale evaluation compared to `demo.py`, supporting batched inference, multi-GPU inference, torch compilation, and skipping video compositions.

An example of running this script with four GPUs is as follows:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4  batch_eval.py duration_s=5 dataset=urmp model=small_44k_sa num_workers=8
```

You may need to update the data paths in `config/eval_data/base.yaml`.
More configuration options can be found in `config/base_config.yaml` and `config/eval_config.yaml`.

## TODO

- [ ] Add on-the-fly segmentation mask generation from user prompts.
