# SVoRT
SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI

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

To generate training data, please download the [CRL atlas](http://crl.med.harvard.edu/research/fetal_brain_atlas/) and [FeTA dataset v2.1](http://neuroimaging.ch/feta), unzip them in ```dataset/```, and run ```preprocessing.py```.

### Modify hyperparameters

The hyperparameters of data simulation and model are stored in ```./src/config/```.

### Run the training script

```python train.py SVoRT```

## Pretrained model

To use the pretrained model, please first download the [pretrain weights](https://zenodo.org/record/7121298#.YzS4R3bML-g).

## Testing

```python test.py SVoRT <path-to-model-checkpoint> <path-to-output-folder>```

## Citation

```
@article{xu2022svort,
  title={SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI},
  author={Xu, Junshen and Moyer, Daniel and Grant, P Ellen and Golland, Polina and Iglesias, Juan Eugenio and Adalsteinsson, Elfar},
  journal={arXiv preprint arXiv:2206.10802},
  year={2022}
}
```

## Contact

For questions, please send an email to junshen@mit.edu
