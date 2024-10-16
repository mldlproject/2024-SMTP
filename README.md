# 2024-SMTP
### Requirements 

```
PyTorch >= 2.0.0
torch-gemetric >= 1.3.2
rdkit
scikit-learn
```

### Usage example
Download all dataset from this [link](https://drive.google.com/file/d/1h0Hdi4fwiivN2IjSHIxv6xjqtUYkyrOo/view)

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

