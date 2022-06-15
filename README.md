# Quick introduction_to_CFR
## CFR?
### Papers
- [Shalit et al., 2017] Shalit, Uri, Fredrik D. Johansson, and David Sontag. "Estimating individual treatment effect: generalization bounds and algorithms." International Conference on Machine Learning. PMLR, 2017
- [Johansson et al., 2016] Johansson, Fredrik, Uri Shalit, and David Sontag. "Learning representations for counterfactual inference." International conference on machine learning. PMLR, 2016.

The following two github repositories were used as references.
- [cfrnet] https://github.com/clinicalml/cfrnet 
- [SC-CFR] https://github.com/koh-t/SC-CFR

The former([cfrnet]) is the official implementation of the original; it is implemented in TensorFlow, but the algorithm inside was used as a reference. The latter([SC-CFR]) is implemented in PyTorch, the same as mine. The architecture of the model is different, but I used many of the class definitions, etc. as reference

## Installation
(TODO: Organize requirements.txt and docker file)
## Usage

```
$ python experiment_run.py
```

If you want to change a hyperparameters:
```
$ python experiment_run.py -m alpha=0,0.1,0.01,0.001,0.0001,1,100,10000,100000,1000000,10000000,100000000,1000000000,10000000000,100000000000 split_outnet=True,False
```

To check results:
```
$ mlflow ui
```
