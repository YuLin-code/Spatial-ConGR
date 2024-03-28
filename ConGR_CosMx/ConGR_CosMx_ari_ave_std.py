import numpy as np
import torch
import warnings
import time
import argparse
import os
import random
import torch.utils.data as Data
import pandas as pd
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='the mode of CL')

    # original
    parser.add_argument('-check', type=int, nargs='+', default=[5], help='the number of epcoh to check.')
    parser.add_argument('-batch_size', type=int, nargs='+', default=[64], help='the batch_size in contrastive learning.')
    parser.add_argument('-model_epoch', type=int, nargs='+', default=[1], help='the epoch_train in contrastive learning.')
    parser.add_argument('-model_lr', type=float, nargs='+', default=[1e-3], help='the learning_rate in contrastive learning.')
    parser.add_argument('-temperature_cl', type=float, nargs='+', default=[0.1], help='the temperature_coefficient in contrastive learning.')

    # model
    parser.add_argument('-transform_opt', type=str, nargs='+', default=['raw'], help='the data normalization for raw rna input.')
    parser.add_argument('-image_arch', type=str, nargs='+', default=['resnet18'], help='the image data encoder in contrastive learning.')
    parser.add_argument('-cl_emb_dim', type=int, nargs='+', default=[128], help='the embedding output dimension in contrastive learning.')
    parser.add_argument('-gae_hidden1_dim', type=int, nargs='+', default=[256], help='the dimension number of the first layer in GAE.')
    parser.add_argument('-gae_hidden2_dim', type=int, nargs='+', default=[128], help='the dimension number of the second layer in GAE.')
    parser.add_argument('-gcn_encoder_dropout', type=float, nargs='+', default=[0.0], help='the dropout rate in GCN.')
    parser.add_argument('-gcn_decoder_dropout', type=float, nargs='+', default=[0.0], help='the dropout rate in GCN.')
    parser.add_argument('-k_graph', type=int, nargs='+', default=[4], help='parameter k in spatial KNN graph.')
    parser.add_argument('-use_exp_nor', type=str, nargs='+', default=['01'], help='the data normalization for use expression in rna data.')
    parser.add_argument('-device', type=str, nargs='+', default=['cpu'], help='the device used for training model: cpu or cuda.')
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
    check = args.check[0]
    batch_size = args.batch_size[0]
    model_epoch = args.model_epoch[0]
    model_lr = args.model_lr[0]
    temperature_cl = args.temperature_cl[0]
    print('check',check)
    print('batch_size',batch_size)
    print('model_epoch',model_epoch)
    print('model_lr',model_lr)
    print('temperature_cl',temperature_cl)

    #load model setting
    transform_opt = args.transform_opt[0]
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
    select_gene_num = args.select_gene_num[0]
    print('transform_opt',transform_opt)
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
    print('select_gene_num',select_gene_num)
    
    #os
    result_tmp_path = './ari_ave_std_result/'
    if not os.path.exists(result_tmp_path):
        os.makedirs(result_tmp_path)

    #folder
    output_path = './gcn_k'+str(k_graph)+'_'+image_arch+'_simclr_'+transform_opt+'_nor'+use_exp_nor+'_batch'+str(batch_size)+\
                  '_epoch'+str(model_epoch)+'_lr'+str(model_lr)+'_r2i'+str(w_rna_image)+'_gae'+str(w_rna_gae)+'/'
                  
    sample_name_list = []
    for sample_num in range(30):
        sample_name_current = 'fov'+str(sample_num+1)
        sample_name_list.append(sample_name_current)
        print('sample_name_current',sample_name_current)
    
    #-------------------training process-----------------#
    learning_name ='g1L'+str(gae_hidden1_dim)+'_g2L'+str(gae_hidden2_dim)+'_k'+str(k_graph)+'_'+transform_opt +'_nor'+use_exp_nor+'_embD' + str(cl_emb_dim)+'_temp'+str(temperature_cl)+'_'+str(image_arch)+ \
                   '_batch'+str(batch_size)+'_lr'+str(model_lr)+'_epoch'+str(model_epoch)+'_r2i'+str(w_rna_image)+'_gae'+str(w_rna_gae)
    print('learning_name ', learning_name)

    emb1_norm_cl1_ari_list = []
    emb1_norm_cl2_ari_list = []
    emb1_cl1_ari_list = []
    emb2_cl1_ari_list = []
    emb1_cl2_ari_list = []
    emb2_cl2_ari_list = []
    
    count = 0
    for len in range(len(sample_name_list)):
        sample_name = sample_name_list[len]
        count = count + 1
        result_path = output_path+sample_name+'/cluster_result/'
        emb1_norm_cl1_result_file = result_path+learning_name+'_emb1_norm_cl1_ari.csv'
        emb1_norm_cl2_result_file = result_path+learning_name+'_emb1_norm_cl2_ari.csv'
        emb1_cl1_result_file = result_path+learning_name+'_emb1_cl1_ari.csv'
        emb2_cl1_result_file = result_path+learning_name+'_emb2_cl1_ari.csv'
        emb1_cl2_result_file = result_path+learning_name+'_emb1_cl2_ari.csv'
        emb2_cl2_result_file = result_path+learning_name+'_emb2_cl2_ari.csv'

        emb1_norm_cl1_result_pd = pd.read_csv(emb1_norm_cl1_result_file,index_col=0)
        emb1_norm_cl2_result_pd = pd.read_csv(emb1_norm_cl2_result_file,index_col=0)
        emb1_cl1_result_pd = pd.read_csv(emb1_cl1_result_file,index_col=0)
        emb2_cl1_result_pd = pd.read_csv(emb2_cl1_result_file,index_col=0)
        emb1_cl2_result_pd = pd.read_csv(emb1_cl2_result_file,index_col=0)
        emb2_cl2_result_pd = pd.read_csv(emb2_cl2_result_file,index_col=0)
        
        print('emb1_norm_cl1_result_pd',emb1_norm_cl1_result_pd)
        print('emb1_norm_cl2_result_pd',emb1_norm_cl2_result_pd)
        print('emb1_cl1_result_pd',emb1_cl1_result_pd)
        print('emb2_cl1_result_pd',emb2_cl1_result_pd)
        print('emb1_cl2_result_pd',emb1_cl2_result_pd)
        print('emb2_cl2_result_pd',emb2_cl2_result_pd)
        
        if check <= 15:
            epoch_loc = 0
        else:
            epoch_loc = int(check/1)-1
            #epoch_loc = int(check/5)-1
        #epoch_loc = int(check/1)-1
        print('epoch_loc',epoch_loc)
        
        emb1_norm_cl1_result_np = emb1_norm_cl1_result_pd.values
        emb1_norm_cl2_result_np = emb1_norm_cl2_result_pd.values
        emb1_cl1_result_np = emb1_cl1_result_pd.values
        emb2_cl1_result_np = emb2_cl1_result_pd.values
        emb1_cl2_result_np = emb1_cl2_result_pd.values
        emb2_cl2_result_np = emb2_cl2_result_pd.values

        emb1_norm_cl1_result_get = emb1_norm_cl1_result_np[epoch_loc]
        emb1_norm_cl2_result_get = emb1_norm_cl2_result_np[epoch_loc]
        emb1_cl1_result_get = emb1_cl1_result_np[epoch_loc]
        emb2_cl1_result_get = emb2_cl1_result_np[epoch_loc]
        emb1_cl2_result_get = emb1_cl2_result_np[epoch_loc]
        emb2_cl2_result_get = emb2_cl2_result_np[epoch_loc]
        
        print('emb1_norm_cl1_result_get',emb1_norm_cl1_result_get)
        print('emb1_norm_cl2_result_get',emb1_norm_cl2_result_get)
        print('emb1_cl1_result_get',emb1_cl1_result_get)
        print('emb2_cl1_result_get',emb2_cl1_result_get)
        print('emb1_cl2_result_get',emb1_cl2_result_get)
        print('emb2_cl2_result_get',emb2_cl2_result_get)
        
        emb1_norm_cl1_ari_list.append(emb1_norm_cl1_result_get)
        emb1_norm_cl2_ari_list.append(emb1_norm_cl2_result_get)
        emb1_cl1_ari_list.append(emb1_cl1_result_get)
        emb2_cl1_ari_list.append(emb2_cl1_result_get)
        emb1_cl2_ari_list.append(emb1_cl2_result_get)
        emb2_cl2_ari_list.append(emb2_cl2_result_get)
        
    emb1_norm_cl1_ari_list_np = np.array(emb1_norm_cl1_ari_list).reshape(-1,1)
    emb1_norm_cl2_ari_list_np = np.array(emb1_norm_cl2_ari_list).reshape(-1,1)
    emb1_cl1_ari_list_np = np.array(emb1_cl1_ari_list).reshape(-1,1)
    emb2_cl1_ari_list_np = np.array(emb2_cl1_ari_list).reshape(-1,1)
    emb1_cl2_ari_list_np = np.array(emb1_cl2_ari_list).reshape(-1,1)
    emb2_cl2_ari_list_np = np.array(emb2_cl2_ari_list).reshape(-1,1)

    emb1_norm_cl1_ari_list_np_ave = np.mean(emb1_norm_cl1_ari_list_np)
    emb1_norm_cl1_ari_list_np_std0 = np.std(emb1_norm_cl1_ari_list_np)
    emb1_norm_cl1_ari_list_np_std1 = np.std(emb1_norm_cl1_ari_list_np,ddof=1)
    emb1_norm_cl1_ari_list_ave_std_np = np.array([emb1_norm_cl1_ari_list_np_ave,emb1_norm_cl1_ari_list_np_std0,emb1_norm_cl1_ari_list_np_std1]).reshape(-1,1)
    print('emb1_norm_cl1_ari_list_ave_std_np',emb1_norm_cl1_ari_list_ave_std_np)

    emb1_norm_cl2_ari_list_np_ave = np.mean(emb1_norm_cl2_ari_list_np)
    emb1_norm_cl2_ari_list_np_std0 = np.std(emb1_norm_cl2_ari_list_np)
    emb1_norm_cl2_ari_list_np_std1 = np.std(emb1_norm_cl2_ari_list_np,ddof=1)
    emb1_norm_cl2_ari_list_ave_std_np = np.array([emb1_norm_cl2_ari_list_np_ave,emb1_norm_cl2_ari_list_np_std0,emb1_norm_cl2_ari_list_np_std1]).reshape(-1,1)
    print('emb1_norm_cl2_ari_list_ave_std_np',emb1_norm_cl2_ari_list_ave_std_np)
    
    emb1_cl1_ari_list_np_ave = np.mean(emb1_cl1_ari_list_np)
    emb1_cl1_ari_list_np_std0 = np.std(emb1_cl1_ari_list_np)
    emb1_cl1_ari_list_np_std1 = np.std(emb1_cl1_ari_list_np,ddof=1)
    emb1_cl1_ari_list_ave_std_np = np.array([emb1_cl1_ari_list_np_ave,emb1_cl1_ari_list_np_std0,emb1_cl1_ari_list_np_std1]).reshape(-1,1)
    print('emb1_cl1_ari_list_ave_std_np',emb1_cl1_ari_list_ave_std_np)

    emb2_cl1_ari_list_np_ave = np.mean(emb2_cl1_ari_list_np)
    emb2_cl1_ari_list_np_std0 = np.std(emb2_cl1_ari_list_np)
    emb2_cl1_ari_list_np_std1 = np.std(emb2_cl1_ari_list_np,ddof=1)
    emb2_cl1_ari_list_ave_std_np = np.array([emb2_cl1_ari_list_np_ave,emb2_cl1_ari_list_np_std0,emb2_cl1_ari_list_np_std1]).reshape(-1,1)
    print('emb2_cl1_ari_list_ave_std_np',emb2_cl1_ari_list_ave_std_np)

    emb1_cl2_ari_list_np_ave = np.mean(emb1_cl2_ari_list_np)
    emb1_cl2_ari_list_np_std0 = np.std(emb1_cl2_ari_list_np)
    emb1_cl2_ari_list_np_std1 = np.std(emb1_cl2_ari_list_np,ddof=1)
    emb1_cl2_ari_list_ave_std_np = np.array([emb1_cl2_ari_list_np_ave,emb1_cl2_ari_list_np_std0,emb1_cl2_ari_list_np_std1]).reshape(-1,1)
    print('emb1_cl2_ari_list_ave_std_np',emb1_cl2_ari_list_ave_std_np)

    emb2_cl2_ari_list_np_ave = np.mean(emb2_cl2_ari_list_np)
    emb2_cl2_ari_list_np_std0 = np.std(emb2_cl2_ari_list_np)
    emb2_cl2_ari_list_np_std1 = np.std(emb2_cl2_ari_list_np,ddof=1)
    emb2_cl2_ari_list_ave_std_np = np.array([emb2_cl2_ari_list_np_ave,emb2_cl2_ari_list_np_std0,emb2_cl2_ari_list_np_std1]).reshape(-1,1)
    print('emb2_cl2_ari_list_ave_std_np',emb2_cl2_ari_list_ave_std_np)
    
    ave_std_index_list = ['ave','std0','std1']
    ari_ave_std_final_index_list = sample_name_list+ave_std_index_list

    emb1_norm_cl1_ari_list_final_np = np.vstack((emb1_norm_cl1_ari_list_np,emb1_norm_cl1_ari_list_ave_std_np))
    emb1_norm_cl1_ari_list_ave_std_pd = pd.DataFrame(emb1_norm_cl1_ari_list_final_np,columns = ['values'],index=ari_ave_std_final_index_list)
    emb1_norm_cl1_ari_list_ave_std_pd.to_csv(result_tmp_path+learning_name+'_check'+str(check)+'_emb1_norm_cl1_ari_ave_std.csv')  

    emb1_norm_cl2_ari_list_final_np = np.vstack((emb1_norm_cl2_ari_list_np,emb1_norm_cl2_ari_list_ave_std_np))
    emb1_norm_cl2_ari_list_ave_std_pd = pd.DataFrame(emb1_norm_cl2_ari_list_final_np,columns = ['values'],index=ari_ave_std_final_index_list)
    emb1_norm_cl2_ari_list_ave_std_pd.to_csv(result_tmp_path+learning_name+'_check'+str(check)+'_emb1_norm_cl2_ari_ave_std.csv')  
    
    emb1_cl1_ari_list_final_np = np.vstack((emb1_cl1_ari_list_np,emb1_cl1_ari_list_ave_std_np))
    emb1_cl1_ari_list_ave_std_pd = pd.DataFrame(emb1_cl1_ari_list_final_np,columns = ['values'],index=ari_ave_std_final_index_list)
    emb1_cl1_ari_list_ave_std_pd.to_csv(result_tmp_path+learning_name+'_check'+str(check)+'_emb1_cl1_ari_ave_std.csv')  
    
    emb2_cl1_ari_list_final_np = np.vstack((emb2_cl1_ari_list_np,emb2_cl1_ari_list_ave_std_np))
    emb2_cl1_ari_list_ave_std_pd = pd.DataFrame(emb2_cl1_ari_list_final_np,columns = ['values'],index=ari_ave_std_final_index_list)
    emb2_cl1_ari_list_ave_std_pd.to_csv(result_tmp_path+learning_name+'_check'+str(check)+'_emb2_cl1_ari_ave_std.csv')
    
    emb1_cl2_ari_list_final_np = np.vstack((emb1_cl2_ari_list_np,emb1_cl2_ari_list_ave_std_np))
    emb1_cl2_ari_list_ave_std_pd = pd.DataFrame(emb1_cl2_ari_list_final_np,columns = ['values'],index=ari_ave_std_final_index_list)
    emb1_cl2_ari_list_ave_std_pd.to_csv(result_tmp_path+learning_name+'_check'+str(check)+'_emb1_cl2_ari_ave_std.csv') 

    emb2_cl2_ari_list_final_np = np.vstack((emb2_cl2_ari_list_np,emb2_cl2_ari_list_ave_std_np))
    emb2_cl2_ari_list_ave_std_pd = pd.DataFrame(emb2_cl2_ari_list_final_np,columns = ['values'],index=ari_ave_std_final_index_list)
    emb2_cl2_ari_list_ave_std_pd.to_csv(result_tmp_path+learning_name+'_check'+str(check)+'_emb2_cl2_ari_ave_std.csv')