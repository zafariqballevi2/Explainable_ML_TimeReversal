# Explainable_ML_TimeReversal

### Pretraining: 
   ```bash
    python -m Main_pretraining --fold_v 0 --daata chirp1 --encoder rnm --seeds 1 --ws 1200 --nw 1 --wsize 1200 --convsize 238400 --epp 2 --tp 1200 --samples 830 --l_ptr F --attr_alg IG
```


### Downstream training:
  ```bash
  python -m Main_downstreamtraining --fold_v 0 --daata chirp1 --encoder rnm --seeds 1 --ws 140 --nw 1 --wsize 140 --convsize 2400 --epp 2 --tp 140 --samples 311 --l_ptr T --attr_alg IG
  ```

### Description of Arguments:

 ```bash
  --fold_v: Number of fold
  --daata: Pass the data in the shape of:  (Subjects, components, time points)
  --encoder: Model used for training (rnm: The proposed model, cnn: wholeMILC)
  --ws: Window shift (Should be equal to time points
  --wsize: window size ( Should be equal to time points
  --convsize: Convolution size (Based on the input dimensions)
  --epp: Number of epochs
  --tp: time points
  --samples: Number of subjects
  --l_ptr: T (with pretraining), F (w/o pretraining)
  --attr_alg: Attribution Algorithm (IG : Integrated Gradients, GS : Gradient SHAP)
```
