import numpy as np
import pandas as pd
import torch
import warnings
import time
import argparse
import os
from CL_data_preprocess import *
import random
import torch.utils.data as Data
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
warnings.filterwarnings("ignore")
from sklearn import decomposition

def parse_args():
    parser = argparse.ArgumentParser(description='the embedding of PCA method')

    # original
    parser.add_argument('-sample', type=str, nargs='+', default=['151507'], help='the sample name.')
    parser.add_argument('-transform_opt', type=str, nargs='+', default=['raw'], help='the data normalization for raw rna input.')

    args = parser.parse_args()
    return args

def seed_random_torch(seed):
    random.seed(seed)                        
    torch.manual_seed(seed)                  
    np.random.seed(seed)                     
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.cuda.manual_seed(seed)             
    torch.cuda.manual_seed_all(seed)         
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    
    #random setting
    seed_random_torch(300)

    #load original setting
    args = parse_args()
    sample = args.sample[0]
    transform_opt = args.transform_opt[0]
    print('sample',sample)
    print('transform_opt',transform_opt)

    #pca dim set
    emb_pca_dim = 3
    
    #folder
    output_path = '../Dataset/Human_Brain/original_pre_transformed_'+transform_opt+'_'+str(emb_pca_dim)+'D_emb_pca/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    #os
    h5_path = '../Dataset/Human_Brain/original_data_folder/'+ sample +'/filtered_feature_bc_matrix.h5'
    spatial_path = '../Dataset/Human_Brain/original_data_folder/' + sample + '/spatial/tissue_positions_list.csv'
    scale_factor_path = '../Dataset/Human_Brain/original_data_folder/' + sample +'/spatial/scalefactors_json.json'
    image_path = '../Dataset/Human_Brain/image_data_folder/'
    metadata_path = '../Dataset/Human_Brain/meta_data_folder/metaData_brain_16_coords/'

    #load spatial data
    adata , spatial_all = load_data(h5_path, spatial_path, scale_factor_path, transform_opt)

    #spatial data preprocessing
    coords, use_expression_np, ground_truth_label= generate_coords_use_exp(adata, sample, metadata_path)
    #coords, use_expression_np = generate_coords_use_exp_no_label(adata)

    use_expression_np_nor = normalization_min_max(use_expression_np)

    #pca features
    pca_emb = decomposition.PCA(n_components=emb_pca_dim,random_state=300)           #random_state=0
    pca_emb_for_rgb = pca_emb.fit_transform(use_expression_np_nor)

    #embedding save
    emb_col_index_list = []
    for emb_num in range(3):
        emb_col_index_list.append('embedding'+str(emb_num)) 
    
    pca_emb_for_rgb_pd = pd.DataFrame(pca_emb_for_rgb,columns=emb_col_index_list,index=adata.obs.index)    
    pca_emb_for_rgb_pd.to_csv(output_path+'/'+sample+'_original_pre_transformed_'+transform_opt+'_'+str(emb_pca_dim)+'D_pca.csv')



