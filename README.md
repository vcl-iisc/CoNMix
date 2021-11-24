# Source-Free Multi-Target Domain Adaptation
Official pytorch implementation for **Knowledge Distillation based Source-Free Multi-Target Domain Adaptation**.

## Dataset Prepration:
- Please manually download the datasets [Office](https://www.dropbox.com/sh/vja4cdimm0k2um3/AACCKNKV8-HVbEZDPDCyAyf_a?dl=0), [Office-Home](https://www.dropbox.com/sh/vja4cdimm0k2um3/AACCKNKV8-HVbEZDPDCyAyf_a?dl=0), PACS, DomainNet from the official websites, and modify the path of images in each '.txt' under the folder './data/'.
- For downloading DomainNet run `sh final_scripts/download_domain_net.sh`. Manually extract zip and keep directory structure as mentioned in [Dataset directory](#Dataset-directory)

### Dataset directory
<details>
  <summary>Click to see full directory tree</summary>

```
   data
    ├── domain_net
    │   ├── clipart
    │   ├── clipart.txt
    │   ├── infograph
    │   ├── infograph.txt
    │   ├── painting
    │   ├── painting.txt
    │   ├── quickdraw
    │   ├── quickdraw.txt
    │   ├── real
    │   ├── real.txt
    │   ├── sketch
    │   └── sketch.txt
    ├── office
    │   ├── amazon
    │   ├── amazon.txt
    │   ├── dslr
    │   ├── dslr.txt
    │   ├── webcam
    │   └── webcam.txt
    ├── office-home
    │   ├── Art
    │   ├── Art.txt
    │   ├── Clipart
    │   ├── Clipart.txt
    │   ├── Product
    │   ├── Product.txt
    │   ├── Real_World
    │   └── RealWorld.txt
    ├── office_home_mixed
    │   ├── Art_Clipart_Product
    │   ├── Art_Clipart_Product.txt
    │   ├── Art_Clipart_Real_World
    │   ├── Art_Clipart_Real_World.txt
    │   ├── Art_Product_Real_World
    │   ├── Art_Product_Real_World.txt
    │   ├── Clipart_Product_Real_World
    │   └── Clipart_Product_Real_World.txt
    └── pacs
        ├── art_painting
        ├── art_painting.txt
        ├── cartoon
        ├── cartoon.txt
        ├── __MACOSX
        ├── photo
        ├── photo.txt
        ├── sketch
        └── sketch.txt
```
</details>


## Training

Install the dependencies and run scripts.

### Prerequisites:

- See [requirements.txt](requirements.txt)
- Install dependencies using `pip3 install -r requirements.txt`

### Prepare pretrain model
We choose R50-ViT-B_16 as our backbone.
```sh class:"lineNo"
# Download pretrained R50-ViT-B_16
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz 
mkdir -p ./model/vit_checkpoint/imagenet21k 
mv R50+ViT-B_16.npz ./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

### Stage 1: Source only Training

```sh
# Change parameters for different dataset
sh final_scripts/1_image_source.sh
```

### Stage 2: STDA training
```sh
# Change parameters for different dataset
# Manually set each STDA source and target
sh final_scripts/2_STDA.sh
```

### Stage 3: KD MTDA training
 ```sh
# Change parameters for different dataset
# Manually set each source
sh final_scripts/3_KD_MTDA.sh
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

## Contributers
- [Rohit Lal](https://rohitlal.net) 
- [Amandeep Kumar](https://github.com/VIROBO-15)
- [Vikash Kumar](https://github.com/vikash0837)

## Code Reference

- [ViT](https://github.com/jeonsworld/ViT-pytorch)
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [SHOT](https://github.com/tim-learn/SHOT)