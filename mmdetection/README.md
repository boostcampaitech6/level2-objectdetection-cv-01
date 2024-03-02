# MMDetection
Official Repo : https://github.com/open-mmlab/mmdetection
## How to install environment
```
pip install openmim
pip install mmcv==2.1.0 mmengine==0.10.2
```
## How to use
1. Make config file
    MMDetection provides a variety of config files. You must modify the config.py of model to option you want to use. We created custom config files comprising the model, dataset, scheduler, runtime in `mmdetection/custom_config`.
2. Train
    ```
    python tools/train.py [config file path]
    ``` 
3. Inference
    ```
    python tools/test.py [config file path]
    ```
4. Make confusion matrix
    ```
    python tools/analysis_tools/confusion_matrix.py
    ```
