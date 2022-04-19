# Advanced-DL-Final-Project

This repo contains our final project code based on CutMix and MixUp papers.

## Instructions

### Train our tests on CIFAR100:
In my_test parameter choose the test you wish to run: 1, 2 or 3.

-	Test 1: choose random rows or columns and change them to rows or columns from another image (CutMix).
-	Test 2: choose random pixels and change them to pixels from another image (CutMix). 
-	Test 3: choose random pixels (different randomization method from Test 2) and change them to pixels from another image (CutMix). 

```
python train.py \
--net_type resnet \
--dataset cifar100 \
--depth 18 \
--alpha 240 \
--batch_size 64 \
--lr 0.25 \
--expname test_1 \
--epochs 300 \
--beta 1.0 \
--my_test 1 \  
--no-verbose
```
### Test model on CIFAR100 corrupted:

```
python test.py \
--net_type resnet \
--dataset cifar100C \
--depth 18 \
--batch_size 64 \
--pretrained ${MODEL_PATH} 
```
or use our notebook [test_models.ipynb](https://github.com/adidigit/advanced-dl-final-project/blob/main/test_models.ipynb)


