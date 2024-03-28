# Tutorial of Spatial-ConGR

## A contrastive learning approach to integrate spatial transcriptomics and H&E data for single-cell analysis

ConGcR and ConGaR, which are contrastive learning-based models, can effectively integrate spatial multi-omics data with gene expression, spatial location and morphology image to model embedded representations for more accurate spatial tissue architecture identification.

## System Requirements:

ConGcR and ConGaR were implemented on a computing server with 2.2 GHz, 144 cores CPU, 503 GB RAM and one NVIDIA TU102 [TITAN RTX] GPU under an ubuntu 18.04 operating system.

### OS Requirements: 

ConGcR and ConGaR can run on both Linux and Windows. The package has been tested on the Linux system.

### Install Spatial-ConGR from Github:

```bash
git clone https://github.com/YuLin-code/Spatial-ConGR.git
cd Spatial-ConGR
```

### Python Dependencies: 

ConGcR and ConGaR depend on the Python (3.6+) scientific stack and python virutal environment with conda (<https://anaconda.org/>) is recommended.

```shell
conda create -n Spatial_ConGR python=3.8
conda activate Spatial_ConGR
pip install -r requirements.txt
```

## Examples:

### 1. Benchmark Methods

Directly clustering, adding and concatenating the preprocessed features of two modalities were used as the original clustering and baseline spatial multi-omics data integration methods in this study. The high resolution H&E images of chicken heart can be downloaded from Cornell Box: https://cornell.box.com/s/u1itpy7vl9zbmwo8vd60wd6rrpik007m, please put them into the directory of 'Dataset/Chicken_Heart/chicken_heart_spatial_RNAseq_raw/'. Run the following command to generate the baseline spatial domain identification results and we take '151507' and 'D4' as the example samples with raw gene expression and gray-scale image.

```bash
cd Benchmark_Methods
python Benchmark_Method_HB.py -sample 151507 -transform_opt raw -image_type gray
python Benchmark_Method_CH.py -sample D4 -transform_opt raw -image_type gray 
```

### 2. ConGcR and ConGaR Models in Human Brain Case

To validate the proposed models of ConGcR and ConGaR, 16 human brain spatial multi-omics samples were used. We take the examples of ConGcR and ConGaR with raw gene expression, batch_size is 64, model_epoch is 5 and other hyper-parameter settings are default in human brain dataset case. If you want to use GPU to run the models, please set the hyper-parameter of device as cuda and configurate the environment as the Pytorch with CUDA enabled. Here we use the following settings on sample 151507 to demo purposes:

- **transform_opt** defines the normalization method on gene expression input.
- **batch_size** defines the number of batch size in the model training process.
- **model_epoch** defines the number of epoch in the model training process.
- **w_rna_image** defines the weight of simclr loss using the features of two modalities in the final loss.
- **w_rna_gae** defines the weight of gae loss using gene expression in the final loss.
- **device** defines the device that is used for training model.

```bash
cd ..
cd ConGR_Human_Brain
python ConGR_Train_HB.py -sample 151507 -transform_opt raw -batch_size 64 -model_epoch 5 -w_rna_image 1 
python ConGR_Train_HB.py -sample 151507 -transform_opt raw -batch_size 64 -model_epoch 5 -w_rna_image 1 -w_rna_gae 100 
```

### 3. ConGcR and ConGaR Models in Chicken Heart Case

To evaluate the model availability on new test data, four chicken heart spatial multi-omics samples were used. We take the examples of ConGcR and ConGaR with logCPM normalized gene expression, batch_size is 128, model_epoch is 15 and other hyper-parameter settings are default in chicken heart dataset case. If you want to use GPU to run the models, please set the hyper-parameter of device as cuda and configurate the environment as the Pytorch with CUDA enabled. Here we use the following settings on sample D4 to demo purposes:

```bash
cd ..
cd ConGR_Chicken_Heart
python ConGR_Train_CH.py -sample D4 -transform_opt logcpm -batch_size 128 -model_epoch 15 -w_rna_image 1 
python ConGR_Train_CH.py -sample D4 -transform_opt logcpm -batch_size 128 -model_epoch 15 -w_rna_image 1 -w_rna_gae 100 
```

### 4. Embedding Dimensional Reduction and RGB Generation

Three embedding dimensional reduction methods of PCA, t-SNE and UMAP were applied to transform the embeddings into three-dimensional features, that were used to generate RGB images referencing the generation method in RESEPT. Here we use the application with PCA method in preprocessed features and ConGR embeddings of raw gene expression on sample 151509 to demo purposes:

```bash
cd ..
cd Dimensional_Reduction_and_Generate_RGB
python original_pre_emb_pca.py -sample 151509 -transform_opt raw
python ConGR_emb_pca.py -sample 151509 -check 5 -transform_opt raw -batch_size 64 -model_epoch 5 -w_rna_image 1
python ConGR_emb_pca.py -sample 151509 -check 5 -transform_opt raw -batch_size 64 -model_epoch 5 -w_rna_image 1 -w_rna_gae 100
python transform_embedding_to_RGB.py -sample 151509 -transformDimMethod pca -transform_opt raw
```

### 5. Image Quality Evaluation

For selecting the most effective dimensional reduction method, we evaluated the RGB image quality between model embedding and original preprocessed embedding by the assessment metrics of PSNR, SSIM and MSE. Run the following command lines to obtain the comparison results of sample 151509: 

```bash
python Image_Quality_Evaluation.py -sample 151509 -transform_opt raw
```

### 6. ConGcR and ConGaR Models in Breast Tumor Case

To enhance model usage scenario, eight HER2-positive breast tumor spatial multi-omics samples of Spatial Transcriptomics (ST) technology were used. We take the examples of ConGcR and ConGaR with the same hyper-parameter settings in chicken heart dataset case. If you want to use GPU to run the models, please set the hyper-parameter of device as cuda and configurate the environment as the Pytorch with CUDA enabled.  Here we use the following settings on sample D1 to demo purposes:

```bash
cd ..
cd ConGR_ST
python ConGR_Train_ST.py -sample D1 -transform_opt raw -batch_size 128 -model_epoch 15 -w_rna_image 1 
python ConGR_Train_ST.py -sample D1 -transform_opt raw -batch_size 128 -model_epoch 15 -w_rna_image 1 -w_rna_gae 100 
```

### 7. DEG Analysis

To further illustrate the superiority of the proposed model in biological explorations, we conducted the DEG analysis based on the learned labels of ConGcR. Run the following command lines to obtain the DEG results of sample D1: 

```bash
python ConGR_DEG.py -sample D1 -transform_opt raw
```

### 8. ConGcR and ConGaR Models in Human Lung Case

To enhance model usage scenario, thirty human lung spatial multi-omics samples of CosMx Spatially Molecular Imaging (SMI) technology were used. We take the examples of ConGcR and ConGaR with the same hyper-parameter settings in chicken heart dataset case. If you want to use GPU to run the models, please set the hyper-parameter of device as cuda and configurate the environment as the Pytorch with CUDA enabled.  Here we use the following settings on sample fov1 to demo purposes:

```bash
cd ..
cd ConGR_CosMx
python ConGR_Train_CosMx.py -sample fov1 -transform_opt raw -batch_size 128 -model_epoch 15 -w_rna_image 1 
python ConGR_Train_CosMx.py -sample fov1 -transform_opt raw -batch_size 128 -model_epoch 15 -w_rna_image 1 -w_rna_gae 100
```

## Document Description:

In the respective file paths, we have the following files.

- ***_CL_baseline_ari.csv**:    Ari comparison results of original clustering methods and the benchmark methods of integrating spatial multi-omics data. 

- ***_emb2_cl1.csv**:    The gene expression embedding of GCN encoder before mapped into the shared space in contrastive learning.

- ***_emb2_cl1_ari.csv**:    The saved results of tissue architecture identification during training process in the proposed models of ConGcR or ConGaR.

- ***_original_transformed_raw_3D_pca.csv**:    Three dimensional embedding generated by PCA method using original preprocessed features with raw gene expression. 

- ***_emb2_cl1_3D_pca.csv**:    Three dimensional embedding generated by PCA method using the representations of ConGcR or ConGaR with raw gene expression.

- ***_image_quality_evaluation.csv**:    RGB image quality comparison results of using PCA, t-SNE or UMAP methods for transforming different embeddings into three-dimensional features.

- ***_transformed_lowres.png**:    RGB image converted from the corresponding embedding.

- ***_emb2_cl1_deg.csv**:    The saved results of DEG analysis ranked by adjusted p-value using the learned labels of ConGcR.