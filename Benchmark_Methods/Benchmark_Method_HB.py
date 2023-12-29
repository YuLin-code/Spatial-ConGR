import pandas as pd
import numpy as np
import torch
import warnings
import time
import argparse
import os
from method_utils import *
import random
import cv2
warnings.filterwarnings("ignore")
from sklearn import decomposition
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None

def parse_args():
    parser = argparse.ArgumentParser(description='the hyperparameter settings of CL baseline method')
    parser.add_argument('-sample', type=str, nargs='+', default=['151507'], help='which sample to generate: 151507-101510,151669-151676,18-64,2-5,2-8,T4857.')
    parser.add_argument('-transform_opt', type=str, nargs='+', default=['logcpm'], help='normalization method(raw/logcpm)')
    parser.add_argument('-image_type', type=str, nargs='+', default=['gray'], help='cropped image type(gray/rgb)')
    args = parser.parse_args()
    return args

def seed_random(seed):
    random.seed(seed)                        
    torch.manual_seed(seed)                  
    np.random.seed(seed)                     
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.cuda.manual_seed(seed)             
    torch.cuda.manual_seed_all(seed)         
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    
    #load setting
    args = parse_args()
    sample = args.sample[0]
    transform_opt = args.transform_opt[0]
    image_type = args.image_type[0]
    print('sample',sample)
    print('transform_opt',transform_opt)
    print('image_type',image_type)
    
    #random setting
    seed_random(300)
    
    #sample_list
    sample_human_brain_list = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676','18-64','2-5','2-8','T4857']
    sample_chicken_heart_list = ['D4','D7','D10','D14']

    #os
    h5_path = '../Dataset/Human_Brain/original_data_folder/'+ sample +'/filtered_feature_bc_matrix.h5'
    spatial_path = '../Dataset/Human_Brain/original_data_folder/' + sample + '/spatial/tissue_positions_list.csv'
    scale_factor_path = '../Dataset/Human_Brain/original_data_folder/' + sample +'/spatial/scalefactors_json.json'
    image_path = '../Dataset/Human_Brain/image_data_folder/'
    metadata_path = '../Dataset/Human_Brain/meta_data_folder/metaData_brain_16_coords/'
    
    #folder
    output_path = './CL_Baseline_Method_Human_Brain_Result/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tile_path = output_path+'/image_tile/'
    if not os.path.exists(tile_path):
        os.makedirs(tile_path)
    
    #load spatial data
    geneSelectnum = 2000
    adata = load_data(h5_path, spatial_path, scale_factor_path, transform_opt)
    
    #spatial data preprocessing
    #coords, use_expression_np, ground_truth_label= generate_coords_use_exp(adata, sample, metadata_path)
    coords, use_expression_np = generate_coords_use_exp_no_label(adata)
    
    #use expression normalization
    use_expression_np_nor = normalization_min_max(use_expression_np)

    #load image data 
    if sample == '2-5' or sample == '2-8' or sample == '18-64' or sample == 'T4857':
        if image_type == 'gray':
            img_gray = cv2.imread(image_path+sample+'_A1.jpg',0)
        if image_type == 'rgb':
            img_3d = cv2.imread(image_path+sample+'_A1.jpg',1)
    else:
        if image_type == 'gray':
            #img_gray = cv2.imread(image_path+sample+'_12samples_A1.tif',0)
            img_gray = cv2.imread(image_path+sample+'_full_image.tif',0)
        if image_type == 'rgb':
            #img_3d = cv2.imread(image_path+sample+'_12samples_A1.tif',1)
            img_3d = cv2.imread(image_path+sample+'_full_image.tif',1)
    #crop data
    radius = int(0.5 *  adata.uns['fiducial_diameter_fullres'] + 1)
    print('radius init is',radius)
    radius = radius * 1
    print('radius used is',radius)
    
    #patch_target_size
    patch_target_size = radius*2
    
    #spot_row_col
    spot_row_in_fullres=adata.obs['pxl_col_in_fullres'].values
    spot_col_in_fullres=adata.obs['pxl_row_in_fullres'].values
    
    #image data preprocessing
    if image_type == 'gray':
        cropped_img, cropped_img_nor = image_data_prepare_for_original_gray(img_gray, radius, spot_row_in_fullres, spot_col_in_fullres)
    if image_type == 'rgb':
        adata_barcode = adata.obs.index
        cropped_img, cropped_img_nor = image_data_prepare_tile(img_3d, radius, spot_row_in_fullres, spot_col_in_fullres, patch_target_size, adata_barcode, tile_path)
        #cropped_img, cropped_img_nor = image_data_prepare_for_original_3d(img_3d, radius, spot_row_in_fullres, spot_col_in_fullres)

    #pca features
    pca_img = decomposition.PCA(n_components=geneSelectnum)           #random_state=0
    cropped_img_nor_pca = pca_img.fit_transform(cropped_img_nor)

    #scale
    cropped_img_nor_pca_scaled_min_max = normalization_min_max(cropped_img_nor_pca)

    #add or concatenate output    rna nor + image scale 
    rna_nor_img_scale_add_np = use_expression_np_nor + cropped_img_nor_pca_scaled_min_max
    rna_nor_img_scale_concatenate_np = np.concatenate((use_expression_np_nor,cropped_img_nor_pca_scaled_min_max),axis=1)

    #embedding save
    emb_col_index_list = []
    for emb_num in range(geneSelectnum):
        emb_col_index_list.append('embedding'+str(emb_num)) 
    
    #DataFrame
    use_expression_nor_pd = pd.DataFrame(use_expression_np_nor,columns=emb_col_index_list,index=adata.obs.index)
    cropped_img_nor_pca_pd = pd.DataFrame(cropped_img_nor_pca,columns=emb_col_index_list,index=adata.obs.index)   
    
    #obtain ari
    original_top_gene_nor_ari = Embedding_Kmeans_Result(use_expression_np_nor,sample,adata,metadata_path)

    #scale
    cropped_img_nor_pca_scaled_min_max_ari = Embedding_Kmeans_Result(cropped_img_nor_pca_scaled_min_max,sample,adata,metadata_path)
    rna_nor_img_scale_add_ari = Embedding_Kmeans_Result(rna_nor_img_scale_add_np,sample,adata,metadata_path)
    rna_nor_img_scale_concatenate_ari = Embedding_Kmeans_Result(rna_nor_img_scale_concatenate_np,sample,adata,metadata_path)
    

    ari_rna_img_result_list = [original_top_gene_nor_ari, cropped_img_nor_pca_scaled_min_max_ari, rna_nor_img_scale_add_ari, rna_nor_img_scale_concatenate_ari]
    ari_rna_img_result_list_np = np.array(ari_rna_img_result_list).reshape(1,-1)
    
    if image_type == 'gray':
        result_pd_index = ['Gene_Gray_IMG_ARI']
    if image_type == 'rgb':
        result_pd_index = ['Gene_RGB_IMG_ARI']
    
    #save ari result
    ari_rna_img_result_pd = pd.DataFrame(ari_rna_img_result_list_np,columns=['original_top_gene_nor_ari', 'cropped_img_nor_pca_scaled_min_max_ari','rna_nor_img_scale_add_ari','rna_nor_img_scale_concatenate_ari'],index=result_pd_index)
    ari_rna_img_result_pd.to_csv(output_path+'/'+sample+'_rna_normalization_'+transform_opt+'_image_type_'+image_type+'_human_brain_CL_baseline_ari.csv')
