# 2024-SMTP
### Requirements 

```
PyTorch >= 2.0.0
torch-gemetric >= 1.3.2
rdkit
scikit-learn
```

### Usage example

**All Data for training Encoder**
```sh
cd ./dataset/raw/dataset.csv
```

**Encode data**
```sh
cd ./loader.py
```

**Masking data**
```sh
cd ./util.py
```
**Training STP-BERT**
```sh
python ./training_stp_bert.py
```

**Extract Features**
```sh
python ./extract_feature.py
```

