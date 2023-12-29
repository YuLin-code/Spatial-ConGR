import numpy as np
import pandas as pd
import torch
import warnings
import time
import argparse
import os
from CL_data_preprocess import load_data, generate_coords_use_exp, generate_coords_use_exp_no_label, load_image_data, image_data_prepare_tile, normalization_min_max
import random
import torch.utils.data as Data
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
warnings.filterwarnings("ignore")
from sklearn import decomposition

def parse_args():
    parser = argparse.ArgumentParser(description='the mode of CL')

    # original
    parser.add_argument('-check', type=int, nargs='+', default=[5], help='the number of epoch to check.')
    parser.add_argument('-sample', type=str, nargs='+', default=['151507'], help='the sample name.')
    parser.add_argument('-batch_size', type=int, nargs='+', default=[64], help='the batch_size in contrastive learning.')
    parser.add_argument('-model_epoch', type=int, nargs='+', default=[1], help='the epoch_train in contrastive learning.')
    parser.add_argument('-model_lr', type=float, nargs='+', default=[1e-3], help='the learning_rate in contrastive learning.')
    parser.add_argument('-temperature_cl', type=float, nargs='+', default=[0.1], help='the temperature_coefficient in contrastive learning.')

    # model
    parser.add_argument('-transform_opt', type=str, nargs='+', default=['raw'], help='the data normalization for raw rna input.')
    parser.add_argument('-patch_target_size', type=int, nargs='+', default=[100], help='the target size for image cropped patch.')
    parser.add_argument('-image_arch', type=str, nargs='+', default=['resnet18'], help='the image data encoder in contrastive learning.')
    parser.add_argument('-cl_emb_dim', type=int, nargs='+', default=[128], help='the embedding output dimension in contrastive learning.')
    parser.add_argument('-gae_hidden1_dim', type=int, nargs='+', default=[512], help='the dimension number of the first layer in GAE.')
    parser.add_argument('-gae_hidden2_dim', type=int, nargs='+', default=[128], help='the dimension number of the second layer in GAE.')
    parser.add_argument('-gcn_encoder_dropout', type=float, nargs='+', default=[0.0], help='the dropout rate in GCN.')
    parser.add_argument('-gcn_decoder_dropout', type=float, nargs='+', default=[0.0], help='the dropout rate in GCN.')
    parser.add_argument('-k_graph', type=int, nargs='+', default=[8], help='parameter k in spatial KNN graph.')
    parser.add_argument('-use_exp_nor', type=str, nargs='+', default=['01'], help='the data normalization for use expression in rna data.')
    parser.add_argument('-device', type=str, nargs='+', default=['cpu'], help='the device used for training model.')
    
    # loss
    parser.add_argument('-w_rna_image', type=float, nargs='+', default=[1.0], help='the weight of simclr_rna_image in the final loss.')
    parser.add_argument('-w_rna_gae', type=float, nargs='+', default=[0.0], help='the weight of gae_rna in the final loss.')

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
    check = args.check[0]
    sample = args.sample[0]
    batch_size = args.batch_size[0]
    model_epoch = args.model_epoch[0]
    model_lr = args.model_lr[0]
    temperature_cl = args.temperature_cl[0]
    print('epoch check is ',check)
    print('sample',sample)
    print('batch_size',batch_size)
    print('model_epoch',model_epoch)
    print('model_lr',model_lr)
    print('temperature_cl',temperature_cl)

    #load model setting
    transform_opt = args.transform_opt[0]
    patch_target_size = args.patch_target_size[0]
    image_arch = args.image_arch[0]
    cl_emb_dim = args.cl_emb_dim[0]
    gae_hidden1_dim = args.gae_hidden1_dim[0]
    gae_hidden2_dim = args.gae_hidden2_dim[0]
    gcn_encoder_dropout = args.gcn_encoder_dropout[0]
    gcn_decoder_dropout = args.gcn_decoder_dropout[0]
    k_graph = args.k_graph[0]
    use_exp_nor = args.use_exp_nor[0]
    device = args.device[0]
    w_rna_image = args.w_rna_image[0]
    w_rna_gae = args.w_rna_gae[0]
    print('transform_opt',transform_opt)
    print('patch_target_size',patch_target_size)
    print('image_arch',image_arch)
    print('cl_emb_dim',cl_emb_dim)
    print('gae_hidden1_dim',gae_hidden1_dim)
    print('gae_hidden2_dim',gae_hidden2_dim)
    print('gcn_encoder_dropout',gcn_encoder_dropout)
    print('gcn_decoder_dropout',gcn_decoder_dropout)
    print('k_graph',k_graph)
    print('use_exp_nor',use_exp_nor)
    print('device',device)
    print('w_rna_image',w_rna_image)
    print('w_rna_gae',w_rna_gae)
    
    #pca dim set
    emb_pca_dim = 3
    
    #os
    h5_path = '../Dataset/Human_Brain/original_data_folder/'+ sample +'/filtered_feature_bc_matrix.h5'
    spatial_path = '../Dataset/Human_Brain/original_data_folder/' + sample + '/spatial/tissue_positions_list.csv'
    scale_factor_path = '../Dataset/Human_Brain/original_data_folder/' + sample +'/spatial/scalefactors_json.json'
    image_path = '../Dataset/Human_Brain/image_data/'
    metadata_path = '../Dataset/Human_Brain/meta_data_folder/metaData_brain_16_coords/'

    #folder
    output_path = '../Dataset/Human_Brain/ConGR_emb_server/'

    #adata obtain
    adata , spatial_all = load_data(h5_path, spatial_path, scale_factor_path, transform_opt)

    #-------------------training process-----------------#
    learning_name ='g1L'+str(gae_hidden1_dim)+'_g2L'+str(gae_hidden2_dim)+'_drE'+str(gcn_encoder_dropout)+'_drD'+str(gcn_decoder_dropout)+'_k'+str(k_graph)+'_'+transform_opt +'_nor'+use_exp_nor+'_embD' + str(cl_emb_dim)+'_temp'+str(temperature_cl)+'_'+str(image_arch)+ \
                   '_batch'+str(batch_size)+'_lr'+str(model_lr)+'_epoch'+str(model_epoch)+'_r2i'+str(w_rna_image)+'_gae'+str(w_rna_gae)+'_epoch'+str(check)
    print('learning_name ', learning_name)
    
    embedding_path = output_path+sample+'/embedding/'
    embedding_pca_path = output_path+sample+'/embedding_pca/'
    if not os.path.exists(embedding_pca_path):
        os.makedirs(embedding_pca_path)

    embedding_file = embedding_path+learning_name+'_emb2_cl1.csv'
    embedding_pd = pd.read_csv(embedding_file,index_col=0)
    embedding_np = embedding_pd.values
    
    #pca features
    pca_emb = decomposition.PCA(n_components=emb_pca_dim,random_state=300)           #random_state=0
    pca_emb_for_rgb = pca_emb.fit_transform(embedding_np)
    
    #embedding save
    emb_col_index_list = []
    for emb_num in range(3):
        emb_col_index_list.append('embedding'+str(emb_num)) 
    
    assert (embedding_pd.index == adata.obs.index).all()
    pca_emb_for_rgb_pd = pd.DataFrame(pca_emb_for_rgb,columns=emb_col_index_list,index=adata.obs.index)    
    pca_emb_for_rgb_pd.to_csv(embedding_pca_path+'/'+sample+'_'+learning_name+'_emb2_cl1_'+str(emb_pca_dim)+'D_pca.csv')

