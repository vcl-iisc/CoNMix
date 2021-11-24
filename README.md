# Source-Free Multi-Target Domain Adaptation
Official pytorch implementation for **Knowledge Distillation based Source-Free Multi-Target Domain Adaptation**.

## Dataset Prepration:
- Please manually download the dataset [Office-Home](https://www.dropbox.com/sh/vja4cdimm0k2um3/AACCKNKV8-HVbEZDPDCyAyf_a?dl=0),  from the official websites, and modify the path of images in each '.txt' under the folder './data/'.

### Dataset directory
<details>
  <summary>Click to see full directory tree</summary>

```
   data

    ├── office-home
        ├── Art
        ├── Art.txt
        ├── Clipart
        ├── Clipart.txt
        ├── Product
        ├── Product.txt
        ├── Real_World
        └── RealWorld.txt

```
</details>


## Training

Install the dependencies and run scripts.

### Prerequisites:

- See [requirements.sh](scripts/requirements.sh)
- Install dependencies using `pip3 install -r scripts/requirements.sh`

### Stage 1: Source only Training

```sh
sh scripts/source_train.sh
```

### Stage 2: STDA training
```sh
sh scripts/STDA.sh
```

### Stage 3: KD MTDA training
 ```sh
sh scripts/MTDA.sh
 ```

### Testing 

For testing any model use the [test_model_acc.py](test_model_acc.py) code. There are two function
- `multi_domain_avg_acc()` : Gives average acc across all domains (equal weight to all domains irrespective of images)
- `test_model()` : Testing on single domain

Changes to be done in code
- txt files
- saved model path
- model type
- domain/list of domain to be used in function `multi_domain_avg_acc()` or `test_model()`
- update `bottleneck_dim` variable as per your training model


## Code Reference

- [TransUNet](https://github.com/Beckschen/TransUNet)
- [SHOT](https://github.com/tim-learn/SHOT)