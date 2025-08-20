---
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: label
    dtype:
      class_label:
        names:
          '0': '0'
          '1': '1'
          '2': '2'
          '3': '3'
          '4': '4'
          '5': '5'
          '6': '6'
          '7': '7'
          '8': '8'
          '9': '9'
  splits:
  - name: train
    num_bytes: 18835641.6
    num_examples: 2700
  - name: test
    num_bytes: 2084899.0
    num_examples: 300
  download_size: 19632369
  dataset_size: 20920540.6
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
