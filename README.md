# Prior Knowledge Guided Network for Video Anomaly Detection

## 1. Dependencies
```
python==3.8.8
pytorch==1.9.0
mmcv-full==1.4.8
mmdet==2.23.0
scikit-learn==0.24.1
pyyaml==5.4.1
```
## 2. Usage
### 2.1 Preprocess
Please follow the [instructions](./preprocess/README.md) to prepare the training and testing data.
Extract STCs by running:
```python
$ python get_STC.py [--dataset_name] \
					[--dataset_path] \
					[--mode]
```
E.g., for the test set of CUHK Avenue:
```python
$ python get_STC.py --dataset_name 'avenue' \
					--dataset_path './preprocess/avenue' \
					--mode 'test'
```

### 2.2 Train
Start training by running:
```python
$ python train.py [--config_path]
```
E.g., for the CUHK Avenue dataset:
```python
$ python train.py --config_path './config/avenue.yaml'
```
### 2.3 Evaluation
To evaluation the anomaly detection performance of the saved model, run:
```python
$ python eval.py [--config_path] \
				 [--model_path]
```
E.g., for the CUHK Avenue dataset:
```python
$ python eval.py --config_path './config/avenue.yaml' \
				 --model_path './model/avenue_pred_9350.pt'
```
