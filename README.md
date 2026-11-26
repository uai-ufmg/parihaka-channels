# Parihaka Channels

Official repo for the **Parihaka-Channels: An Annotated Real-World Dataset for Channels Segmentation using Seismic Horizons and Spectral Decomposition** paper: \[*Link*\]

This project provides a framework for the training and evaluation of deep learning models applied to the segmentation of channels within seismic data. We include different architectures, loss functions, metrics, optimizers, and learning rate schedulers.


## :wrench: Requirements

```bash
pip install -r requirements.txt
```


## :open_file_folder: Dataset

The Parihaka-Channels dataset: \[*Link*\]

The developed annotation tool: \[*Link*\]

The provided dataset includes extracted 2D sections (inlines, crosslines, timeslices) and horizon slices from Parihaka-3D, as well the projected 2D annotations and spectral decomposition. Each sample is sliced in half, with its first half (side A) being used for training and the second half (side B) being used for testing. To separate data splits and point to data paths, the script expects CSV files with relative paths for individual numpy samples such as the example in `data/csv_files/parihaka_sample_test.csv`:

| dataset_name | relative_path |
| -------- | ------- |
| parihaka | path/to/sample_001 |
| parihaka | path/to/sample_002 |
| parihaka | path/to/sample_003 |
| parihaka | path/to/sample_004 |
| parihaka | path/to/sample_005 |

The script searches for files matching the pattern `*_seismic.npy` for amplitude data and `*_label.py` for annotations by default, but these can be changed through the dataset config parameters `attributes` and `label_name` respectively. A list of seismic attributes can be passed to the model provided that files with the matching suffixes are available.

The expected directory structure should look like:

```text
<dataset_dir>/
├── sample_001_label.npy
├── sample_001_seismic.npy
├── sample_001_lowspectral.npy
├── sample_001_midspectral.npy
├── sample_001_highspectral.npy
├── sample_002_label.npy
├── sample_002_seismic.npy
└── ...
```


## :bulb: Training

Experiments are organized through YAML files, which define parameters such as the model, batch size, loss function, and optimizer. An example configuration can be found at `configs/example_config.yaml`.

To train a model from scratch, run a command such as the following:

```bash
python train.py --config path/to/config.yaml --exp_name example_experiment
```


## :bar_chart: Inference

To test and generate predictions, swap `train.py` for `test.py` but keep the same `config` and `exp_name` used during training:

```bash
python test.py --config path/to/config.yaml --exp_name example_experiment
```

To avoid training from scratch, model weights for our best performing model (SegFormer) for both amplitude and spectral decomposition data types are available in the \[*dataset link*\]. We also provide example predictions in `data/predictions` (to be added :smile:).


## :pushpin: Citing

The Parihaka-3D dataset was provided by **New Zealand Petroleum & Minerals (NZPM)** and is available through the [SEG Open Data Repository](https://wiki.seg.org/wiki/Parihaka-3D). Please refer to the original dataset source for the applicable terms of use and citation requirements.