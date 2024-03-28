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
def Validate_CL_Embedding_COSMX(model_epoch, learning_name, sample, adata, embedding_path, meta_path, cluster_result_path, validate_model_epoch):

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
        emb1_cl1_ari_current, emb1_cl1_label_save_current =  Embedding_Kmeans_Result_COSMX(emb1_cl1_np,sample,adata,meta_path)
        emb1_cl2_ari_current, emb1_cl2_label_save_current =  Embedding_Kmeans_Result_COSMX(emb1_cl2_np,sample,adata,meta_path)
        emb2_cl1_ari_current, emb2_cl1_label_save_current =  Embedding_Kmeans_Result_COSMX(emb2_cl1_np,sample,adata,meta_path)
        emb2_cl2_ari_current, emb2_cl2_label_save_current =  Embedding_Kmeans_Result_COSMX(emb2_cl2_np,sample,adata,meta_path)
        emb1_norm_cl1_ari_current, emb1_norm_cl1_label_save_current =  Embedding_Kmeans_Result_COSMX(emb1_cl1_norm_np,sample,adata,meta_path)
        emb1_norm_cl2_ari_current, emb1_norm_cl2_label_save_current =  Embedding_Kmeans_Result_COSMX(emb1_cl2_norm_np,sample,adata,meta_path)
        
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
    
    #embedding_label_pd_save
    emb1_cl1_label_save_current.to_csv(cluster_result_path+'/'+learning_name+'_emb1_cl1_label.csv')
    emb1_cl2_label_save_current.to_csv(cluster_result_path+'/'+learning_name+'_emb1_cl2_label.csv')
    emb2_cl1_label_save_current.to_csv(cluster_result_path+'/'+learning_name+'_emb2_cl1_label.csv')
    emb2_cl2_label_save_current.to_csv(cluster_result_path+'/'+learning_name+'_emb2_cl2_label.csv')
    emb1_norm_cl1_label_save_current.to_csv(cluster_result_path+'/'+learning_name+'_emb1_norm_cl1_label.csv')
    emb1_norm_cl2_label_save_current.to_csv(cluster_result_path+'/'+learning_name+'_emb1_norm_cl2_label.csv')
    
    return print('CL embedding validation finished!')

#embedding clustering
#fov1  7 #immune  #lymphoid structure  #myeloid-enriched stroma  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov2  6 #immune  #lymphoid structure  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov3  6 #immune  #myeloid-enriched stroma  #neutrophils  #stroma  #tumor interior  #tumor-stroma boundary
#fov4  6 #lymphoid structure  #myeloid-enriched stroma  #neutrophils  #stroma  #tumor interior  #tumor-stroma boundary
#fov5  6 #lymphoid structure  #myeloid-enriched stroma  #neutrophils  #stroma  #tumor interior  #tumor-stroma boundary
#fov6  5 #lymphoid structure  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov7  5 #lymphoid structure  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov8  6 #immune  #myeloid-enriched stroma  #neutrophils  #stroma  #tumor interior  #tumor-stroma boundary
#fov9  4 #lymphoid structure  #myeloid-enriched stroma  #stroma  #tumor-stroma boundary
#fov10 7 #lymphoid structure  #myeloid-enriched stroma  #neutrophils  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov11 6 #immune  #lymphoid structure  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov12 7 #lymphoid structure  #myeloid-enriched stroma  #neutrophils  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov13 5 #myeloid-enriched stroma  #neutrophils  #stroma  #tumor interior  #tumor-stroma boundary
#fov14 3 #lymphoid structure  #myeloid-enriched stroma  #stroma
#fov15 7 #lymphoid structure  #macrophages  #myeloid-enriched stroma  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov16 6 #immune  #lymphoid structure  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov17 6 #myeloid-enriched stroma  #neutrophils  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov18 4 #lymphoid structure  #myeloid-enriched stroma  #plasmablast-enriched stroma  #stroma
#fov19 3 #lymphoid structure  #myeloid-enriched stroma  #stroma
#fov20 3 #lymphoid structure  #myeloid-enriched stroma  #stroma
#fov21 3 #lymphoid structure  #plasmablast-enriched stroma  #stroma
#fov22 6 #immune  #lymphoid structure  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov23 3 #lymphoid structure  #myeloid-enriched stroma  #stroma
#fov24 2 #myeloid-enriched stroma  #stroma  
#fov25 3 #lymphoid structure  #myeloid-enriched stroma  #stroma
#fov26 6 #lymphoid structure  #neutrophils  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov27 4 #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov28 8 #immune  #lymphoid structure  #myeloid-enriched stroma  #neutrophils  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov29 6 #myeloid-enriched stroma  #neutrophils  #plasmablast-enriched stroma  #stroma  #tumor interior  #tumor-stroma boundary
#fov30 3 #lymphoid structure  #plasmablast-enriched stroma  #stroma

def Embedding_Kmeans_Result_COSMX(embedding, sample, adata, meta_path):
    #set embedding
    emb_evaluation = embedding
    n_clusters_num = 6     #11 samples
    if sample=='fov28':
        n_clusters_num = 8
    elif sample=='fov1' or sample=='fov10' or sample=='fov12' or sample=='fov15':
        n_clusters_num = 7
    elif sample=='fov6' or sample=='fov7' or sample=='fov13':
        n_clusters_num = 5
    elif sample=='fov9' or sample=='fov18' or sample=='fov27':
        n_clusters_num = 4
    elif sample=='fov14' or sample=='fov19' or sample=='fov20' or sample=='fov21' or sample=='fov23' or sample=='fov25' or sample=='fov30':
        n_clusters_num = 3
    elif sample=='fov24':
        n_clusters_num = 2
    
    print('The current cluster number is ',n_clusters_num)

    #check embedding
    meta_load = pd.read_csv(meta_path+sample+'_cosmx_meta_data.csv', index_col = 0)
    assert all(adata.obs.index == meta_load.index)

    #kmeans
    X = emb_evaluation
    #print('X is',X)
    print('X shape is',X.shape)
    kmeans = KMeans(n_clusters=n_clusters_num, random_state=0).fit(X)
    kmeans_label = kmeans.labels_
    #print('kmeans_label',kmeans_label)
    #print('kmeans_label max is',np.max(kmeans_label))
    kmeans_label_np = kmeans_label.reshape(-1,1)
    kmeans_label_pd = pd.DataFrame(kmeans_label_np,columns=['pre_label'],index=adata.obs.index)
    #print('kmeans_label_pd',kmeans_label_pd)
    metadata_add_pre_label_pd = pd.concat([meta_load,kmeans_label_pd],axis=1)
    print('metadata_add_pre_label_pd',metadata_add_pre_label_pd)   #add
    kmeans_label_pre = metadata_add_pre_label_pd['pre_label'].values
    print('kmeans_label_pre unique num is',len(np.unique(kmeans_label_pre)))
    print('kmeans_label_pre min is',np.min(kmeans_label_pre))
    print('kmeans_label_pre shape is',kmeans_label_pre.shape)
    
    #save label
    label_save_pd = kmeans_label_pd
    
    ground_truth_label_np = np.zeros((metadata_add_pre_label_pd.shape[0],)).astype(int)
    ground_truth_init_np = np.array(metadata_add_pre_label_pd['niche'])
    
    for k_num in range(len(ground_truth_init_np)):
        if ground_truth_init_np[k_num] == 'immune':
            ground_truth_label_np[k_num] = 1
        if ground_truth_init_np[k_num] == 'lymphoid structure':
            ground_truth_label_np[k_num] = 2
        if ground_truth_init_np[k_num] == 'myeloid-enriched stroma':
            ground_truth_label_np[k_num] = 3
        if ground_truth_init_np[k_num] == 'plasmablast-enriched stroma':
            ground_truth_label_np[k_num] = 4
        if ground_truth_init_np[k_num] == 'stroma':
            ground_truth_label_np[k_num] = 5
        if ground_truth_init_np[k_num] == 'tumor interior':
            ground_truth_label_np[k_num] = 6
        if ground_truth_init_np[k_num] == 'tumor-stroma boundary':
            ground_truth_label_np[k_num] = 7
        if ground_truth_init_np[k_num] == 'neutrophils':
            ground_truth_label_np[k_num] = 8
        if ground_truth_init_np[k_num] == 'macrophages':
            ground_truth_label_np[k_num] = 9

    print('ground_truth_label_np unique num is',len(np.unique(ground_truth_label_np)))
    print('ground_truth_label_np min is',np.min(ground_truth_label_np))
    print('ground_truth_label_np shape is', ground_truth_label_np.shape)
        
    assert( len(np.unique(ground_truth_label_np)) == len(np.unique(kmeans_label_pre)) )
    assert( len(np.unique(ground_truth_label_np)) == n_clusters_num )
    
    #obtain ari 
    ari = adjusted_rand_score(kmeans_label_pre , ground_truth_label_np)
    print('ari',ari)
    
    return ari, label_save_pd

#embedding clustering
def Embedding_Kmeans_Result_L7(embedding, sample, adata, metadata_path):
    #set embedding
    emb_evaluation = embedding
    n_clusters_num = 7
    #if sample=='151669' or sample=='151670' or sample=='151671' or sample=='151672':
    #    n_clusters_num = 5
    #if sample=='2-8':
    #    n_clusters_num = 6
        
    print('The current cluster number is ',n_clusters_num)
    
    #check embedding
    #metadata_path = './meta_data_folder/metaData_brain_16_coords/'
    metadata_pd = pd.read_csv(metadata_path+'/'+sample+'_humanBrain_metaData.csv',index_col=0)
    metadata_modify_barcode_pd = metadata_pd.loc[adata.obs.index]
    
    #kmeans
    if sample == '2-5' or sample == '2-8' or sample == '18-64' or sample == 'T4857':
        print('The sample is in the 4_samples!')
        X = emb_evaluation
        #print('X is',X)
        print('X shape is',X.shape)
        kmeans = KMeans(n_clusters=n_clusters_num, random_state=0).fit(X)
        kmeans_label = kmeans.labels_
        #print('kmeans_label',kmeans_label)
        #print('kmeans_label max is',np.max(kmeans_label))
        kmeans_label_np = kmeans_label.reshape(-1,1)
        kmeans_label_pd = pd.DataFrame(kmeans_label_np,columns=['pre_label'],index=adata.obs.index)
        #print('kmeans_label_pd',kmeans_label_pd)
        metadata_modify_barcode_add_pre_label_pd = pd.concat([metadata_modify_barcode_pd,kmeans_label_pd],axis=1)
        #print('metadata_modify_barcode_add_pre_label_pd',metadata_modify_barcode_add_pre_label_pd)
        
        metadata_modify_barcode_add_pre_label_filter_pd = metadata_modify_barcode_add_pre_label_pd.loc[(metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer 1') | (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer 2') | \
                                                                        (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer 3') | (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer 4') | \
                                                                        (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer 5') | (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer 6') | \
                                                                        (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'White matter')]        
        
        kmeans_label_pre_filter = metadata_modify_barcode_add_pre_label_filter_pd['pre_label'].values
        print('kmeans_label_pre_filter',kmeans_label_pre_filter)
        print('kmeans_label_pre_filter unique num is',len(np.unique(kmeans_label_pre_filter)))
        print('kmeans_label_pre_filter min is',np.min(kmeans_label_pre_filter))
        print('kmeans_label_pre_filter shape is',kmeans_label_pre_filter.shape)
        
        ground_truth_label_np = np.zeros((metadata_modify_barcode_add_pre_label_filter_pd.shape[0],)).astype(int)
        ground_truth_init_np = np.array(metadata_modify_barcode_add_pre_label_filter_pd['benmarklabel'])
        
        for k in range(len(ground_truth_init_np)):
            if ground_truth_init_np[k] == 'Layer 1':
                ground_truth_label_np[k] = 1
            if ground_truth_init_np[k] == 'Layer 2':
                ground_truth_label_np[k] = 2
            if ground_truth_init_np[k] == 'Layer 3':
                ground_truth_label_np[k] = 3
            if ground_truth_init_np[k] == 'Layer 4':
                ground_truth_label_np[k] = 4
            if ground_truth_init_np[k] == 'Layer 5':
                ground_truth_label_np[k] = 5
            if ground_truth_init_np[k] == 'Layer 6':
                ground_truth_label_np[k] = 6
            if ground_truth_init_np[k] == 'White matter':
                ground_truth_label_np[k] = 7
        print('ground_truth_label_np', ground_truth_label_np)
        print('ground_truth_label_np unique num is',len(np.unique(ground_truth_label_np)))
        print('ground_truth_label_np min is',np.min(ground_truth_label_np))
        print('ground_truth_label_np shape is', ground_truth_label_np.shape)
        ari = adjusted_rand_score(kmeans_label_pre_filter , ground_truth_label_np)
        
        print('ari',ari)
    
    if sample == '151507' or sample == '151508' or sample == '151509' or sample == '151510' or sample == '151669' or sample == '151670' or sample == '151671' or sample == '151672' or sample == '151673' or sample == '151674' or sample == '151675' or sample == '151676':
        print('The sample is in the 12_samples!')
        X = emb_evaluation
        #print('X is',X)
        print('X shape is',X.shape)
        kmeans = KMeans(n_clusters=n_clusters_num, random_state=0).fit(X)
        kmeans_label = kmeans.labels_
        #print('kmeans_label',kmeans_label)
        #print('kmeans_label max is',np.max(kmeans_label))
        kmeans_label_np = kmeans_label.reshape(-1,1)
        kmeans_label_pd = pd.DataFrame(kmeans_label_np,columns=['pre_label'],index=adata.obs.index)
        #print('kmeans_label_pd',kmeans_label_pd)
        metadata_modify_barcode_add_pre_label_pd = pd.concat([metadata_modify_barcode_pd,kmeans_label_pd],axis=1)
        #print('metadata_modify_barcode_add_pre_label_pd',metadata_modify_barcode_add_pre_label_pd)
        
        metadata_modify_barcode_add_pre_label_filter_pd = metadata_modify_barcode_add_pre_label_pd.loc[(metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer1') | (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer2') | \
                                                                        (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer3') | (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer4') | \
                                                                        (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer5') | (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'Layer6') | \
                                                                        (metadata_modify_barcode_add_pre_label_pd['benmarklabel'] == 'WM')]        
    
        kmeans_label_pre_filter = metadata_modify_barcode_add_pre_label_filter_pd['pre_label'].values
        print('kmeans_label_pre_filter',kmeans_label_pre_filter) 
        print('kmeans_label_pre_filter unique num is',len(np.unique(kmeans_label_pre_filter)))
        print('kmeans_label_pre_filter min is',np.min(kmeans_label_pre_filter))
        print('kmeans_label_pre_filter shape is',kmeans_label_pre_filter.shape)
    
        ground_truth_label_np = np.zeros((metadata_modify_barcode_add_pre_label_filter_pd.shape[0],)).astype(int)
        ground_truth_init_np = np.array(metadata_modify_barcode_add_pre_label_filter_pd['benmarklabel'])
    
        for k in range(len(ground_truth_init_np)):
            if ground_truth_init_np[k] == 'Layer1':
                ground_truth_label_np[k] = 1
            if ground_truth_init_np[k] == 'Layer2':
                ground_truth_label_np[k] = 2
            if ground_truth_init_np[k] == 'Layer3':
                ground_truth_label_np[k] = 3
            if ground_truth_init_np[k] == 'Layer4':
                ground_truth_label_np[k] = 4
            if ground_truth_init_np[k] == 'Layer5':
                ground_truth_label_np[k] = 5
            if ground_truth_init_np[k] == 'Layer6':
                ground_truth_label_np[k] = 6
            if ground_truth_init_np[k] == 'WM':
                ground_truth_label_np[k] = 7
        print('ground_truth_label_np', ground_truth_label_np)
        print('ground_truth_label_np unique num is',len(np.unique(ground_truth_label_np)))
        print('ground_truth_label_np min is',np.min(ground_truth_label_np))
        print('ground_truth_label_np shape is', ground_truth_label_np.shape)
        ari = adjusted_rand_score(kmeans_label_pre_filter , ground_truth_label_np)
        print('ari',ari)

    return ari
