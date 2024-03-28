import numpy as np
import torch
import warnings
import time
import argparse
import os
from CL_embedding import Generate_CL_Embedding
from CL_data_preprocess import load_data_st, generate_coords_use_exp_st, load_image_data_st, image_data_prepare_tile_st, normalization_min_max
from util_function import Validate_CL_Embedding_ST
import random
import torch.utils.data as Data
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='the mode of CL')

    # original
    parser.add_argument('-sample', type=str, nargs='+', default=['A1'], help='which sample to generate: A1,B1,C1,D1,E1,F1,G2,H1.')
    parser.add_argument('-batch_size', type=int, nargs='+', default=[64], help='the batch_size in contrastive learning.')
    parser.add_argument('-model_epoch', type=int, nargs='+', default=[1], help='the epoch_train in contrastive learning.')
    parser.add_argument('-model_lr', type=float, nargs='+', default=[1e-3], help='the learning_rate in contrastive learning.')
    parser.add_argument('-temperature_cl', type=float, nargs='+', default=[0.1], help='the temperature_coefficient in contrastive learning.')

    # model
    parser.add_argument('-transform_opt', type=str, nargs='+', default=['raw'], help='the data normalization for raw rna input.')
    parser.add_argument('-patch_target_size', type=int, nargs='+', default=[100], help='the target size for image cropped patch.')
    parser.add_argument('-image_arch', type=str, nargs='+', default=['resnet18'], help='the image data encoder in contrastive learning.')
    parser.add_argument('-cl_emb_dim', type=int, nargs='+', default=[128], help='the embedding output dimension in contrastive learning.')
    parser.add_argument('-gae_hidden1_dim', type=int, nargs='+', default=[256], help='the dimension number of the first layer in GAE.')
    parser.add_argument('-gae_hidden2_dim', type=int, nargs='+', default=[128], help='the dimension number of the second layer in GAE.')
    parser.add_argument('-gcn_encoder_dropout', type=float, nargs='+', default=[0.0], help='the dropout rate in GCN.')
    parser.add_argument('-gcn_decoder_dropout', type=float, nargs='+', default=[0.0], help='the dropout rate in GCN.')
    parser.add_argument('-k_graph', type=int, nargs='+', default=[4], help='parameter k in spatial KNN graph.')
    parser.add_argument('-use_exp_nor', type=str, nargs='+', default=['01'], help='the data normalization for use expression in rna data.')
    parser.add_argument('-device', type=str, nargs='+', default=['cpu'], help='the device used for training model: cpu or cuda.')
    parser.add_argument('-crop_diameter', type=int, nargs='+', default=[300], help='the crop diameter for image cropped patch.')
    parser.add_argument('-select_gene_num', type=int, nargs='+', default=[500], help='the filtered high variable gene number in data preprocessing.')
    
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
    sample = args.sample[0]
    batch_size = args.batch_size[0]
    model_epoch = args.model_epoch[0]
    model_lr = args.model_lr[0]
    temperature_cl = args.temperature_cl[0]
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
    crop_diameter = args.crop_diameter[0]
    select_gene_num = args.select_gene_num[0]
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
    print('crop_diameter',crop_diameter)
    print('select_gene_num',select_gene_num)

    #os
    emb_path = '../Dataset/Breast_Tumor/gene_embedding/'
    img_path = '../Dataset/Breast_Tumor/image_data/HE/'
    meta_path = '../Dataset/Breast_Tumor/meta_data/'

    #folder
    output_path = './gcn_k'+str(k_graph)+'_'+image_arch+'_crop'+str(crop_diameter)+'_patch'+str(patch_target_size)+'_simclr_'+transform_opt+'_nor'+use_exp_nor+'_batch'+str(batch_size)+\
                  '_epoch'+str(model_epoch)+'_lr'+str(model_lr)+'_r2i'+str(w_rna_image)+'_gae'+str(w_rna_gae)+'/'+ sample

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_save_path = output_path+'/model_save/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    embedding_path = output_path+'/embedding/'
    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)
    loss_path = output_path+'/loss/'
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    tile_path = output_path+'/image_tile/'
    if not os.path.exists(tile_path):
        os.makedirs(tile_path)
    cluster_result_path = output_path+'/cluster_result/'
    if not os.path.exists(cluster_result_path):
        os.makedirs(cluster_result_path)

    #load spatial data
    adata = load_data_st(emb_path, meta_path, transform_opt, sample)

    #spatial data preprocessing
    coords, use_expression_np = generate_coords_use_exp_st(adata,select_gene_num)

    #normalization for use expression
    if use_exp_nor == '01':
        use_expression_np_nor = normalization_min_max(use_expression_np)
        print('use_expression 01nor is used!')
        print('use_expression_np_nor',use_expression_np_nor)
        print('use_expression_np_nor shape is',use_expression_np_nor.shape)
        print('use_expression_np_nor max is', np.max(use_expression_np_nor))
        print('use_expression_np_nor min is', np.min(use_expression_np_nor))
    else:
        use_expression_np_nor = use_expression_np
        print('original use_expression is used!')
        print('use_expression_np_nor',use_expression_np_nor)
        print('use_expression_np_nor shape is',use_expression_np_nor.shape)
        print('use_expression_np_nor max is', np.max(use_expression_np_nor))
        print('use_expression_np_nor min is', np.min(use_expression_np_nor))
    
    #load image data 
    img_3d, crop_radius, spot_row_in_fullres, spot_col_in_fullres = load_image_data_st(img_path, sample, adata, crop_diameter)
    print('spot_row_in_fullres',spot_row_in_fullres)
    print('spot_col_in_fullres',spot_col_in_fullres)
    
    #image data preprocessing
    adata_loc = adata.obs.index
    cropped_img_3d_nor = image_data_prepare_tile_st(img_3d, crop_radius, spot_row_in_fullres, spot_col_in_fullres, patch_target_size, adata_loc, tile_path)
    
    #batch train setting 
    order_train = np.arange(0,adata.X.A.shape[0],1)
    print('generate order_train for batch is',order_train)
    order_train_tensor = torch.tensor(order_train, dtype=torch.long)
    batch_size_used = batch_size                              #add
    if batch_size==0 or batch_size > adata.X.A.shape[0]:
        batch_size_used = adata.X.A.shape[0]                  #add
    print('batch_size used for training is', batch_size_used)  #add
    
    #generate torch dataloader
    use_expression_tensor = torch.tensor(use_expression_np_nor, dtype=torch.float)
    coords_tensor = torch.tensor(coords, dtype=torch.long)
    image_tensor = torch.tensor(cropped_img_3d_nor, dtype=torch.float)
    #ground_truth_label_tensor = torch.tensor(ground_truth_label, dtype=torch.long)
    #torch_dataset = Data.TensorDataset(use_expression_tensor,coords_tensor,image_tensor,ground_truth_label_tensor,order_train_tensor)
    torch_dataset = Data.TensorDataset(use_expression_tensor,coords_tensor,image_tensor,order_train_tensor)
    print('load data finish')    #generating data remind

    #-------------------training process-----------------#
    learning_name ='g1L'+str(gae_hidden1_dim)+'_g2L'+str(gae_hidden2_dim)+'_k'+str(k_graph)+'_'+transform_opt +'_nor'+use_exp_nor+'_embD' + str(cl_emb_dim)+'_temp'+str(temperature_cl)+'_'+str(image_arch)+ \
                   '_batch'+str(batch_size)+'_lr'+str(model_lr)+'_epoch'+str(model_epoch)+'_r2i'+str(w_rna_image)+'_gae'+str(w_rna_gae)
    print('learning_name ', learning_name)

    #time_computing
    start_time = time.time()
    print("GCN_RESNET_CL. Start Time: %s seconds" %(start_time))
    
    #Generate_CL_Embedding function
    validate_model_epoch = Generate_CL_Embedding(torch_dataset, gae_hidden1_dim, gae_hidden2_dim, gcn_encoder_dropout, gcn_decoder_dropout, k_graph, cl_emb_dim, temperature_cl, image_arch, batch_size_used, model_lr, model_epoch, w_rna_image, w_rna_gae, adata, learning_name, output_path, device)   #batch_size_used
    #validate_model_epoch = 5
    
    #time_computing
    end_time = time.time()
    print("GCN_RESNET_CL. End Time: %s seconds" %(end_time))
    print("GCN_RESNET_CL Done. Total Running Time: %s seconds" %(end_time - start_time))
            
    #Validate_CL_Embedding function
    Validate_CL_Embedding_ST(model_epoch, learning_name, sample, adata, embedding_path, meta_path, cluster_result_path, validate_model_epoch)

