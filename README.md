# SVoRT
SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI

## Resources

- [paper (arxiv)](https://arxiv.org/pdf/2206.10802.pdf)

## Requirements

- python 3.9
- pytorch 1.10
- pyyaml
- scikit-image
- antpy
- scipy
- nibabel

## Todo

- [x] model
- [x] training
- [ ] pretrained weights
- [ ] inference

## Training data

To generate training data, please download the [CRL atlas](http://crl.med.harvard.edu/research/fetal_brain_atlas/) and [FeTA dataset v2.1](http://neuroimaging.ch/feta), unzip them in ```dataset/```, and run ```preprocessing.py```.

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
