import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
import numpy as np

#GAE
def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

#earlystopping
def earlystopping(new_loss, best_loss, es):
    if new_loss<best_loss:
        best_loss = new_loss
        # print('------------new_loss---------------', new_loss)
        # print('------------best_loss---------------', best_loss)
        es = 0
    else:
        es = es + 1
    # print('--------------best_loss---------------', best_loss)
    # print('--------------es---------------', es)
    return best_loss, es

#embedding ari validation
def Validate_CL_chicken_heart_Embedding(model_epoch, learning_name, sample, adata, embedding_path, metadata_path, cluster_result_path, validate_model_epoch):

    #ari list
    emb1_cl1_list = []
    emb1_cl2_list = []
    emb1_cl1_add_list = []
    emb2_cl1_list = []
    emb2_cl2_list = []
    emb2_cl1_add_list = []
    emb1_norm_cl1_list = []
    emb1_norm_cl2_list = []
    emb1_norm_cl1_add_list = []
    
    #epoch_np
    epoch_np = np.linspace(validate_model_epoch,int(model_epoch),int(model_epoch/validate_model_epoch)).astype(int)
    epoch_list = epoch_np.tolist()

    for epoch_num in epoch_list:
        
        #embedding_np 
        emb1_cl1_np = pd.read_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_num)+'_emb1_cl1.csv',index_col=0).values
        emb1_cl2_np = pd.read_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_num)+'_emb1_cl2.csv',index_col=0).values
        emb2_cl1_np = pd.read_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_num)+'_emb2_cl1.csv',index_col=0).values
        emb2_cl2_np = pd.read_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_num)+'_emb2_cl2.csv',index_col=0).values
        emb1_cl1_norm_np = pd.read_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_num)+'_emb1_norm_cl1.csv',index_col=0).values
        emb1_cl2_norm_np = pd.read_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_num)+'_emb1_norm_cl2.csv',index_col=0).values
        
        #embedding_ari 
        emb1_cl1_ari_current =  Embedding_Kmeans_Result_Chicken_Heart(emb1_cl1_np,sample,adata,metadata_path)
        emb1_cl2_ari_current =  Embedding_Kmeans_Result_Chicken_Heart(emb1_cl2_np,sample,adata,metadata_path)
        emb2_cl1_ari_current =  Embedding_Kmeans_Result_Chicken_Heart(emb2_cl1_np,sample,adata,metadata_path)
        emb2_cl2_ari_current =  Embedding_Kmeans_Result_Chicken_Heart(emb2_cl2_np,sample,adata,metadata_path)
        emb1_norm_cl1_ari_current =  Embedding_Kmeans_Result_Chicken_Heart(emb1_cl1_norm_np,sample,adata,metadata_path)
        emb1_norm_cl2_ari_current =  Embedding_Kmeans_Result_Chicken_Heart(emb1_cl2_norm_np,sample,adata,metadata_path)
        
        #embedding_ari_list
        emb1_cl1_list.append(emb1_cl1_ari_current)
        emb1_cl2_list.append(emb1_cl2_ari_current)
        emb2_cl1_list.append(emb2_cl1_ari_current)
        emb2_cl2_list.append(emb2_cl2_ari_current)
        emb1_norm_cl1_list.append(emb1_norm_cl1_ari_current)
        emb1_norm_cl2_list.append(emb1_norm_cl2_ari_current)
    
    #embedding_ari_list_np
    ari_emb1_cl1_list_np = np.array(emb1_cl1_list).reshape(-1,1)
    ari_emb1_cl2_list_np = np.array(emb1_cl2_list).reshape(-1,1)
    ari_emb2_cl1_list_np = np.array(emb2_cl1_list).reshape(-1,1)
    ari_emb2_cl2_list_np = np.array(emb2_cl2_list).reshape(-1,1)
    ari_emb1_norm_cl1_list_np = np.array(emb1_norm_cl1_list).reshape(-1,1)
    ari_emb1_norm_cl2_list_np = np.array(emb1_norm_cl2_list).reshape(-1,1)
    
    #embedding_ari_list_pd_save
    ari_emb1_cl1_pd = pd.DataFrame(ari_emb1_cl1_list_np,columns=['ARI'],index=epoch_list)
    ari_emb1_cl1_pd.to_csv(cluster_result_path+'/'+learning_name+'_emb1_cl1_ari.csv')
    ari_emb1_cl2_pd = pd.DataFrame(ari_emb1_cl2_list_np,columns=['ARI'],index=epoch_list)
    ari_emb1_cl2_pd.to_csv(cluster_result_path+'/'+learning_name+'_emb1_cl2_ari.csv')
    
    ari_emb2_cl1_pd = pd.DataFrame(ari_emb2_cl1_list_np,columns=['ARI'],index=epoch_list)
    ari_emb2_cl1_pd.to_csv(cluster_result_path+'/'+learning_name+'_emb2_cl1_ari.csv')
    ari_emb2_cl2_pd = pd.DataFrame(ari_emb2_cl2_list_np,columns=['ARI'],index=epoch_list)
    ari_emb2_cl2_pd.to_csv(cluster_result_path+'/'+learning_name+'_emb2_cl2_ari.csv')

    ari_emb1_norm_cl1_pd = pd.DataFrame(ari_emb1_norm_cl1_list_np,columns=['ARI'],index=epoch_list)
    ari_emb1_norm_cl1_pd.to_csv(cluster_result_path+'/'+learning_name+'_emb1_norm_cl1_ari.csv')
    ari_emb1_norm_cl2_pd = pd.DataFrame(ari_emb1_norm_cl2_list_np,columns=['ARI'],index=epoch_list)
    ari_emb1_norm_cl2_pd.to_csv(cluster_result_path+'/'+learning_name+'_emb1_norm_cl2_ari.csv')

    return print('CL embedding validation finished!')

#embedding clustering
def Embedding_Kmeans_Result_Chicken_Heart(embedding, sample, adata, metadata_path):
    #setting
    if sample == 'D4':
        n_clusters_num = 5
    elif sample == 'D7':
        n_clusters_num = 7
    elif sample == 'D10':
        n_clusters_num = 7
    else:
        n_clusters_num = 6
        
    print('The current cluster number is ',n_clusters_num)

    #load data and meta
    metadata_pd = pd.read_csv(metadata_path,index_col=0)
    metadata_sample_pd = metadata_pd.loc[metadata_pd['orig.ident']==sample]
    label_sample_np = np.array(metadata_sample_pd['region'])

    sample_index_np = np.array(metadata_sample_pd.index)
    sample_barcode_list = []
    for sample_num in range(len(sample_index_np)):
        sample_barcode_current = sample_index_np[sample_num].split('_')[1]
        sample_barcode_list.append(sample_barcode_current)
    metadata_sample_pd.index = sample_barcode_list
    metadata_modify_barcode_pd = metadata_sample_pd.loc[adata.obs.index]

    ground_truth_init_np = np.array(metadata_modify_barcode_pd['region'])
    ground_truth_label_np = np.zeros((ground_truth_init_np.shape[0],)).astype(int)

    for label_num in range(len(ground_truth_init_np)):
        if ground_truth_init_np[label_num] == 'Atria':
            ground_truth_label_np[label_num] = 1
        if ground_truth_init_np[label_num] == 'Compact LV and inter-ventricular septum' or ground_truth_init_np[label_num] == 'Compact LV and \ninter-ventricular septum':
            ground_truth_label_np[label_num] = 2
        if ground_truth_init_np[label_num] == 'Endothelium':
            ground_truth_label_np[label_num] = 3
        if ground_truth_init_np[label_num] == 'Epicardium':
            ground_truth_label_np[label_num] = 4
        if ground_truth_init_np[label_num] == 'Epicardium- like':
            ground_truth_label_np[label_num] = 5
        if ground_truth_init_np[label_num] == 'Outflow tract':
            ground_truth_label_np[label_num] = 6
        if ground_truth_init_np[label_num] == 'Right ventricle':
            ground_truth_label_np[label_num] = 7
        if ground_truth_init_np[label_num] == 'Trabecular LV and endocardium' or ground_truth_init_np[label_num] == 'Trabecular LV and \nendocardium':
            ground_truth_label_np[label_num] = 8
        if ground_truth_init_np[label_num] == 'Valves':
            ground_truth_label_np[label_num] = 9
        if ground_truth_init_np[label_num] == 'Ventricle':
            ground_truth_label_np[label_num] = 10
    print('ground_truth_label_np unique num is',len(np.unique(ground_truth_label_np)))
    print('ground_truth_label_np shape is', ground_truth_label_np.shape)

    #kmeans
    X = embedding
    print('X shape is',X.shape)
    kmeans = KMeans(n_clusters=n_clusters_num, random_state=0).fit(X)
    kmeans_label = kmeans.labels_
    print('kmeans_label unique num is',len(np.unique(kmeans_label)))
    print('kmeans_label shape is',kmeans_label.shape)
    
    ari = adjusted_rand_score(kmeans_label , ground_truth_label_np)
    print('ari',ari)

    return ari
