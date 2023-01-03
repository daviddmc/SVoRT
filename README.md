# Slice-to-Volume Registration Transformer (SVoRT)

This repo is the official implementation of the paper 'SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI'

## Resources

- Paper ( [Springer](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_1) | [arXiv](https://arxiv.org/abs/2206.10802) )
- Check out [NeSVoR](https://github.com/daviddmc/NeSVoR) for an application of SVoRT

## Requirements

- python 3.9
- pytorch 1.10
- pyyaml
- scikit-image
- antpy
- scipy
- nibabel

## Training from scratch

### Generate training data

To generate training data, please download the [CRL atlas](http://crl.med.harvard.edu/research/fetal_brain_atlas/) and [FeTA dataset v2.1](http://neuroimaging.ch/feta), unzip them in ```dataset/```, and run ```preprocessing.py```. You may also add your own training data (see `RegisteredDataset` in `.src/data/dataset.py`).

### Modify hyperparameters

The hyperparameters of data simulation and model are stored in ```./src/config/```.

### Run the training script

```
python train.py --config ./config/config_SVoRTv2.yaml \
                --output ../results/SVoRTv2
```

## Pretrained model

To use the pretrained model, please first download the [pretrain weights](https://zenodo.org/record/7486938#.Y7Sgn3bMI2w).

## Testing

```
python test.py --config ./config/config_SVoRTv2.yaml \
               --output ../results/SVoRTv2/test_output \
               --checkpoint ../results/SVoRTv2/checkpoint.pt
```

## Citation

```
@inproceedings{xu2022svort,
  title={SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI},
  author={Xu, Junshen and Moyer, Daniel and Grant, P Ellen and Golland, Polina and Iglesias, Juan Eugenio and Adalsteinsson, Elfar},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={3--13},
  year={2022},
  organization={Springer}
}
```

## Contact

For questions, please send an email to junshen@mit.edu
