# GeoNav: Empowering MLLMs with Dual-Scale Geospatial Reasoning for Language-Goal Aerial Navigation

## Abstract

Language-goal aerial navigation requires UAVs to localize targets in the complex outdoors such as urban blocks based on textual instruction. The indoor methods are often hard to scale to urban scenes due to ambiguous objects, limited visual field and spatial reasoning. In this work, we propose **GeoNav**, a multi-modal agent for long-range aerial navigation with geospatial awareness. GeoNav operates in three phases–landmark navigation, target search, and precise localization–mimicking human coarse-to-fine spatial reasoning patterns. To support such reasoning, it dynamically builds dual-scale spatial representations. The first is a global but schematic cognitive map, which fuses prior geographic knowledge and embodied visual cues into a top-down and explicit annotated form. It enables fast navigation to the landmark region via intuitive map-based reasoning. The second is a local but delicate scene graph representing hierarchical spatial relationships between landmarks and objects, utilized for accurate target localization. On top of the structured memory, GeoNav employs a spatial chain-of-thought mechanism to enable MLLMs with efficient and interpretable action-making across stages. On the CityNav benchmark, GeoNav surpasses the current SOTA up to 18.4% in success rate and significantly eliminate navigation error.

<p align="center">
  <img src="figures/framework.png" alt="GeoNav Framework" width="80%">
</p>

## Highlights

- **Dual-Scale Spatial Representation**: Combines a global cognitive map for coarse navigation with a local scene graph for precise localization
- **Three-Phase Navigation**: Landmark Navigation → Target Search → Precise Localization, mimicking human spatial reasoning
- **Spatial Chain-of-Thought**: Enables MLLMs with interpretable and efficient decision-making across navigation stages
- **SOTA Performance**: Surpasses existing methods by up to 18.4% in success rate on CityNav benchmark

## Setup

This code was developed with Python 3.10, PyTorch 2.2.2, and CUDA 11.8 on Ubuntu 22.04.

To set up the environment, create the conda environment and install PyTorch.

```bash
conda create -n geonav python=3.10 &&
conda activate geonav &&
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

Then install Set-of-Marks and its dependencies.

```bash
conda install mpi4py

pip install git+https://github.com/water-cookie/Segment-Everything-Everywhere-All-At-Once.git@package
pip install git+https://github.com/water-cookie/Semantic-SAM.git@package
pip install git+https://github.com/facebookresearch/segment-anything.git

git clone https://github.com/water-cookie/SoM.git  &&
cd SoM/ops && ./make.sh && cd ..  &&
pip install --editable . && cd ..
```

Next, install LLaVA and Grounding DINO.

```bash
pip install git+https://github.com/water-cookie/LLaVA.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

Once LLaVA and Grounding DINO are installed, install the dependencies for GeoNav.

```bash
pip install -r requirements.txt
```

Finally, the weights can be downloaded by running the following script.

```bash
sh scripts/download_weights.sh
```

The downloaded weight files should be organized in the following hierarchy.

```text
GeoNav/
├─ weights/
│  ├─ groundingdino/
│  │  ├─ groundingdino_swinb_cogcoor.pth
│  │  ├─ groundingdino_swint_ogc.pth
│  ├─ mobile_sam/
│  │  ├─ mobile_sam.pt
│  ├─ som/
│  │  ├─ sam_vit_h_4b8939.pth
│  │  ├─ seem_focall_v1.pt
│  │  ├─ swinl_only_sam_many2many.pth
│  ├─ vlnce/
│  │  ├─ ddppo-models/
│  │  │  ├─ gibson-2plus-resnet50.pth
│  │  │  ├─ ...
│  │  ├─ R2R_VLNCE_v1-3_preprocessed/
│  │  │  ├─ embeddings.json.gz
│  │  │   ...
```

## Data Preparation

The dataset can be downloaded with the following script.

```bash
sh scripts/download_data.sh
```

Download [SensatUrban dataset](https://github.com/QingyongHu/SensatUrban?tab=readme-ov-file#4-training-and-evaluation)
and rasterize the point clouds using [CloudCompare](https://www.danielgm.net/cc/) with the grid step size set to `0.1` and the projection direction set to `z`.
Save the RGB data as a `.png` file by setting the active layer to `RGB` and exporting the data with the `Image` button.
The depth image can be saved in the same way by setting the active layer to `Height grid values` and exporting it as a `.tiff` file with the `Raster` button.

The dataset and images should be placed in the directories presented below.

```bash
GeoNav/
├─ data/
│  ├─ cityrefer/
│  │  ├─ objects.json
│  │  ├─ processed_descriptions.json
│  ├─ citynav/
│  │  ├─ citynav_train_seen.json
│  │  ├─ ...
│  ├─ rgbd/
│  │  ├─ birmingham_block_0.png
│  │  ├─ birmingham_block_0.tiff
│  │  ├─ ...
│  ├─ gsam/
│  │  ├─ full_scan_(100, 240, 410).npz
```

## Usage

### GeoNav Agent (Main Method)

Run the following script to evaluate GeoNav on the CityNav benchmark.

```bash
python main_geonav.py \
    --mode eval \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --train_trajectory_type mturk \
    --eval_batch_size 50 \
    --eval_max_timestep 20 \
    --checkpoint checkpoints/data/mgp_mturk.pth \
    --output_dir results/geonav \
    --split test_unseen
```

### Scene Graph Baseline

```bash
python main_scene_graph.py \
    --mode eval \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --train_trajectory_type mturk \
    --eval_batch_size 50 \
    --eval_max_timestep 20 \
    --checkpoint checkpoints/data/mgp_mturk.pth \
    --output_dir results/scene_graph \
    --split test_unseen
```

### Greedy Baseline

```bash
python main_greedy_baseline.py \
    --mode eval \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --train_trajectory_type mturk \
    --eval_batch_size 50 \
    --eval_max_timestep 20 \
    --checkpoint checkpoints/data/mgp_mturk.pth \
    --output_dir results/greedy \
    --split test_unseen
```

### VLLM Deployment (Optional)

For faster inference with local VLLM server:

```bash
# Start VLLM server
bash scripts/start_vllm.sh

# Run with local deployment
python main_geonav.py \
    --mode eval \
    --deployment local \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --train_trajectory_type mturk \
    --eval_batch_size 50 \
    --eval_max_timestep 20 \
    --checkpoint checkpoints/data/mgp_mturk.pth \
    --output_dir results/geonav_local \
    --split test_unseen
```

## Project Structure

```
GeoNav/
├── main_geonav.py              # Main GeoNav agent
├── main_scene_graph.py         # Scene graph baseline
├── main_greedy_baseline.py     # Greedy baseline
├── gsamllavanav/               # Core navigation modules
│   ├── maps/                   # Map representations (cognitive map)
│   ├── dataset/                # Data loading utilities
│   ├── observation/            # Visual observation processing
│   └── ...
├── scenegraphnav/              # Scene graph navigation
│   ├── agent.py                # Navigation agents
│   ├── prompt/                 # Prompt templates
│   └── ...
├── utils/                      # Utility functions
├── scripts/                    # Shell scripts
└── data/                       # Dataset (not included)
```

## Citation

If you find GeoNav useful for your research, please cite our paper:

```bibtex
@article{xu2026geonav,
  title={GeoNav: Empowering MLLMs with dual-scale geospatial reasoning for language-goal aerial navigation},
  author={Xu, Haotian and Hu, Yue and Gao, Chen and Zhu, Zhengqiu and Zhao, Yong and Yin, Quanjun},
  journal={Pattern Recognition},
  pages={113365},
  year={2026},
  publisher={Elsevier}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

We would like to express our gratitude to the authors of the following codebase.

- [CityNav](https://github.com/city-nav/citynav)
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [Set-of-Mark prompting](https://github.com/microsoft/SoM)
- [LLaVA](https://llava-vl.github.io/)
