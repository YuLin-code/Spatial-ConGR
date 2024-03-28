import pandas as pd
import os,json
import numpy as np
import cv2
import scanpy as sc
import scipy.sparse as sp
import torch
import time
from scipy.spatial import distance_matrix, minkowski_distance, distance
import networkx as nx
import warnings
from PIL import Image
import anndata 
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

warnings.filterwarnings("ignore")

def load_data_st(emb_path, meta_path, transform_opt, sample):

    # Read in gene expression and spatial location
    count_load = pd.read_csv(emb_path+sample+'_st_gene_embedding.csv', index_col = 0)
    meta_load = pd.read_csv(meta_path+sample+'_st_meta_data.csv', index_col = 0)
    assert all(count_load.index == meta_load.index)   #add
    print('Data index is ok!!!!!!!!',all(count_load.index == meta_load.index))   #add
    count_load_used = count_load.loc[meta_load.index]
    assert all(count_load_used.index == meta_load.index)
    

    count_load_array = count_load_used.values
    count_load_csr = csr_matrix(count_load_array)
    adata = anndata.AnnData(count_load_csr)
    adata.obs.index = count_load_used.index
    adata.obs.index.name = 'sp_loc'
    adata.obs["array_row"] = meta_load['x_spot']
    adata.obs["array_col"] = meta_load['y_spot']
    adata.obs["pxl_col_in_fullres"] = meta_load['pixel_x_spot']
    adata.obs["pxl_row_in_fullres"] = meta_load['pixel_y_spot']
    adata.var.index = count_load_used.columns
    adata.var_names_make_unique()
    
    # transform optional
    if transform_opt == 'log':
        sc.pp.log1p(adata)
        print('rna data transform is log!')
    elif transform_opt == 'logcpm':
        sc.pp.normalize_total(adata,target_sum=1e4)
        sc.pp.log1p(adata)
        print('rna data transform is logcpm!')
    elif transform_opt == 'raw':
        print('rna data transform is raw!')
    
    print('rna data max is',np.max(adata.X.A))
    return adata

def Label_Kmeans_Result_ST_Check(embedding, sample, adata, meta_path, label_check):
    #set embedding
    emb_evaluation = embedding
    n_clusters_num = 3
    if sample=='G2' or sample=='H1':
        n_clusters_num = 6
    elif sample=='A1':
        n_clusters_num = 5
    elif sample=='B1':
        n_clusters_num = 4
    print('The current cluster number is ',n_clusters_num)
    
    #check embedding
    meta_load = pd.read_csv(meta_path+sample+'_st_meta_data.csv', index_col = 0)
    assert all(adata.obs.index == meta_load.index)

    #kmeans
    X = emb_evaluation
    #print('X is',X)
    print('X shape is',X.shape)
    kmeans = KMeans(n_clusters=n_clusters_num, random_state=0).fit(X)
    kmeans_label = kmeans.labels_
    kmeans_label_np = kmeans_label.reshape(-1,1)
    kmeans_label_pd = pd.DataFrame(kmeans_label_np,columns=['pre_label'],index=adata.obs.index)
    #print('kmeans_label_pd',kmeans_label_pd)
    metadata_add_pre_label_pd = pd.concat([meta_load,kmeans_label_pd],axis=1)
    #print('metadata_add_pre_label_pd',metadata_add_pre_label_pd)   #add
    
    metadata_add_pre_label_filter_pd = metadata_add_pre_label_pd.loc[(metadata_add_pre_label_pd['label'] == 'adipose tissue') | (metadata_add_pre_label_pd['label'] == 'cancer in situ') | \
                                                                    (metadata_add_pre_label_pd['label'] == 'connective tissue') | (metadata_add_pre_label_pd['label'] == 'immune infiltrate') | \
                                                                    (metadata_add_pre_label_pd['label'] == 'invasive cancer') | (metadata_add_pre_label_pd['label'] == 'breast glands')]        
    print('metadata_add_pre_label_filter_pd',metadata_add_pre_label_filter_pd)   #add
    
    kmeans_label_pre_filter = metadata_add_pre_label_filter_pd['pre_label'].values
    #print('kmeans_label_pre_filter',kmeans_label_pre_filter)
    print('kmeans_label_pre_filter unique num is',len(np.unique(kmeans_label_pre_filter)))
    print('kmeans_label_pre_filter min is',np.min(kmeans_label_pre_filter))
    print('kmeans_label_pre_filter shape is',kmeans_label_pre_filter.shape)
    
    #save label
    label_save_pd = metadata_add_pre_label_filter_pd[['label','pre_label']]
    
    ###check part
    label_check_reshape = label_check
    label_save_np = kmeans_label_pre_filter
    #print('label_check_reshape shape is',label_check_reshape.shape)
    #print('label_save_np shape is',label_save_np.shape)
    #print('label_check_reshape type is',type(label_check_reshape))
    #print('label_save_np type is',type(label_save_np))
    #print('label_check_reshape type is',label_check_reshape.dtype)
    #print('label_save_np type is',label_save_np.dtype)
    print('??',(label_check_reshape==label_save_np).all())
    print('label_check is ok!!!')
    ###continue for original    
    
    ground_truth_label_np = np.zeros((metadata_add_pre_label_filter_pd.shape[0],)).astype(int)
    ground_truth_init_np = np.array(metadata_add_pre_label_filter_pd['label'])
    
    for k_num in range(len(ground_truth_init_np)):
        if ground_truth_init_np[k_num] == 'adipose tissue':
            ground_truth_label_np[k_num] = 1
        if ground_truth_init_np[k_num] == 'cancer in situ':
            ground_truth_label_np[k_num] = 2
        if ground_truth_init_np[k_num] == 'connective tissue':
            ground_truth_label_np[k_num] = 3
        if ground_truth_init_np[k_num] == 'immune infiltrate':
            ground_truth_label_np[k_num] = 4
        if ground_truth_init_np[k_num] == 'invasive cancer':
            ground_truth_label_np[k_num] = 5
        if ground_truth_init_np[k_num] == 'breast glands':
            ground_truth_label_np[k_num] = 6
    #print('ground_truth_label_np', ground_truth_label_np)
    print('ground_truth_label_np unique num is',len(np.unique(ground_truth_label_np)))
    print('ground_truth_label_np min is',np.min(ground_truth_label_np))
    print('ground_truth_label_np shape is', ground_truth_label_np.shape)
        
    assert( len(np.unique(ground_truth_label_np)) == len(np.unique(kmeans_label_pre_filter)) )
    
    #obtain ari 
    ari = adjusted_rand_score(kmeans_label_pre_filter , ground_truth_label_np)
    print('ari',ari)
    
    return kmeans_label_pd


def load_data_st_deg(emb_path, embedding_path, cluster_result_path, meta_path, learning_name, adata_tmp, sample, model_epoch): 

    # Read in gene expression and spatial location
    count_load = pd.read_csv(emb_path+sample+'_st_gene_embedding.csv', index_col = 0)
    meta_load = pd.read_csv(meta_path+sample+'_st_meta_data.csv', index_col = 0)
    label_load = pd.read_csv(cluster_result_path+learning_name+'_emb2_cl1_label.csv', index_col = 0)
    emb_load = pd.read_csv(embedding_path+learning_name+'_epoch'+str(model_epoch)+'_emb2_cl1.csv', index_col = 0)
    assert all(count_load.index == meta_load.index)   #add
    #assert all(label_load.index == meta_load.index)   #add
    assert all(emb_load.index == meta_load.index)   #add
    count_load_used = count_load
    count_load_array = count_load_used.values
    count_load_csr = csr_matrix(count_load_array)
    adata = anndata.AnnData(count_load_csr)
    adata.obs.index = count_load_used.index
    adata.obs.index.name = 'sp_loc'
    adata.obs["selected"] = meta_load['selected']
    adata.obs["array_row"] = meta_load['x_spot']
    adata.obs["array_col"] = meta_load['y_spot']
    adata.obs["imagecol"] = meta_load['pixel_x_spot']
    adata.obs["imagerow"] = meta_load['pixel_y_spot']
    adata.var.index = count_load_used.columns
    adata.var['gene_ids'] = count_load_used.columns
    adata.var_names_make_unique()
    adata.obsm['spatial'] = meta_load[['pixel_x_spot','pixel_y_spot']].values

    #check label
    emb_load_np = emb_load.values
    label_check = label_load['pre_label'].values 
    kmeans_label_pd = Label_Kmeans_Result_ST_Check(emb_load_np, sample, adata_tmp, meta_path, label_check)
    adata.obs["congr_kmeans"] = kmeans_label_pd['pre_label'].astype('category')
    
    return adata

def load_image_data_st(img_path, sample, adata, crop_diameter):

    #if sample == '2-5' or sample == '2-8' or sample == '18-64' or sample == 'T4857':
    #    img_3d = cv2.imread(image_path+sample+'_A1.jpg',1)
    #    img_3d = cv2.cvtColor(img_3d, cv2.COLOR_BGR2RGB)
    #else:
    #    #img_3d = cv2.imread(image_path+sample+'_12samples_A1.tif',1)
    #    img_3d = cv2.imread(image_path+sample+'_full_image.tif',1)
    #    img_3d = cv2.cvtColor(img_3d, cv2.COLOR_BGR2RGB)
    
    img_3d = cv2.imread(img_path+sample+'.jpg',1)
    img_3d = cv2.cvtColor(img_3d, cv2.COLOR_BGR2RGB)


    #crop setting
    #crop_radius_init = int(0.5 *  adata.uns['fiducial_diameter_fullres'] + 1)
    #print('crop_radius_init is',crop_radius_init)

    #if crop_radius_init <= 112:
    #    crop_radius = 112
    #else:
    #    crop_radius = crop_radius_init
    crop_radius = crop_diameter/2
    print('crop_radius used is', crop_radius)
    
    spot_row_in_fullres=adata.obs['pxl_row_in_fullres'].values
    spot_col_in_fullres=adata.obs['pxl_col_in_fullres'].values
    
    return img_3d, crop_radius, spot_row_in_fullres, spot_col_in_fullres

def normalization_min_max(inputdata):
    _range = np.max(inputdata) - np.min(inputdata)
    return (inputdata - np.min(inputdata)) / _range


def calculateSpatialMatrix(featureMatrix, distanceType='euclidean', k=6, pruneTag='NA', spatialMatrix=None):
    r"""
    Calculate spatial Matrix directly use X/Y coordinates
    """
    edgeList=[]

    ## Version 2: for each of the cell, calculate dist, save memory
    p_time = time.time()
    for i in np.arange(spatialMatrix.shape[0]):
        #if i%10000==0:                                                                        #original exists but mask for log
        #    print('Start pruning '+str(i)+'th cell, cost '+str(time.time()-p_time)+'s')       #original exists but mask for log
        tmp=spatialMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,spatialMatrix, distanceType)
        # if k == 0, then use all the possible data
        # minus 1 for not exceed the array size
        if k == 0:
            k = spatialMatrix.shape[0]-1
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        #boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
        #boundary = 10 #optional
        #boundary_near = 30 #optional
        for j in np.arange(1,k+1):
            # No prune
            #if distMat[0, res[0][j]] <= boundary:
            #if distMat[0, res[0][j]] <= boundary and distMat[0, res[0][j]] <= boundary_near:
                #edgeList.append((i,res[0][j],1.0))
            edgeList.append((i,res[0][j],1.0))
    return edgeList

# edgeList to edgeDict
def edgeList2edgeDict(edgeList, nodesize):
    graphdict={}
    tdict={}

    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1]=""
        tdict[end2]=""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1]= tmplist

    #check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i]=[]
    return graphdict

def coords_to_adj(zOut, coords, k_graph):
    #SpatialMatrix = preprocessSpatial(coords)
    #SpatialMatrix = torch.from_numpy(SpatialMatrix)
    SpatialMatrix = torch.from_numpy(coords)
    SpatialMatrix = SpatialMatrix.type(torch.FloatTensor)
    prunetype = 'spatialGrid'
    knn_distance = 'euclidean'
    k = k_graph
    #print('knn_graph k is',k)
    pruneTag = 'NA'
    useGAEembedding = True
    adj, edgeList = generateAdj(zOut, graphType=prunetype, para=knn_distance + ':' + str(k) + ':' +pruneTag, adjTag=useGAEembedding, spatialMatrix=SpatialMatrix)
    return adj

def generateAdj(featureMatrix, graphType='spatialGrid', para=None, parallelLimit=0, adjTag=True, spatialMatrix=None):
    """
    Generating edgeList
    """
    edgeList = None
    adj = None

    if graphType == 'spatialGrid':
        # spatial Grid X,Y as the edge
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
            pruneTag = parawords[2]
        edgeList = calculateSpatialMatrix(featureMatrix, distanceType=distanceType, k=k, pruneTag=pruneTag, spatialMatrix=spatialMatrix)
    else:
        print('Should give graphtype')

    if adjTag:
        graphdict = edgeList2edgeDict(edgeList, featureMatrix.shape[0])
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
        
    return adj, edgeList

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # sparse_mx = sparse_mx.tocoo().astype(np.float64)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.DoubleTensor(indices, values, shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = np.arange(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        if ~ismember([idx_i,idx_j],edges_all) and ~ismember([idx_j,idx_i],edges_all):
            val_edges_false.append([idx_i, idx_j])
        else:                                          #original
            # Debug
            print(str(idx_i)+" "+str(idx_j))           #original
            #print(str(idx_i)+" Debug "+str(idx_j))      #modify
            
        # Original:
        # val_edges_false.append([idx_i, idx_j])

    #TODO: temporary disable for ismember function may require huge memory.
    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def sparse_to_COO(adj):
    tmp_coo = sp.coo_matrix(adj)
    values = tmp_coo.data
    indices = np.vstack((tmp_coo.row,tmp_coo.col))
    i = torch.LongTensor(indices)
    # print('indices',indices)
    # print('i',i)
    # v = torch.LongTensor(values)
    # edge_index = torch.sparse_coo_tensor(i,v,tmp_coo.shape)
    return i
    # return edge_index

def graph_processing(train_adj):

    adj_norm = preprocess_graph(train_adj)
    #print('adj_norm',adj_norm)
    #print('adj_norm shape is',adj_norm.shape)
    #print('adj_norm type is',type(adj_norm))
    #n_nodes = adj_norm.shape[0]
    
    #adj preprocessing
    adj_label = train_adj + sp.eye(train_adj.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    pos_weight = float(train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) / train_adj.sum()
    norm = train_adj.shape[0] * train_adj.shape[0] / float((train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) * 2)        
    return adj_norm, adj_label, pos_weight, norm

#-------------------------------test transform tensor list to numpy list--------------------------#
def transform_tensor_to_numpy(x,device):
    if device == 'cuda':
        x_np = x.cpu().detach().numpy()
    else:
        x_np = x.detach().numpy()
    return x_np

#-----------------------------------RNA data preprocess--------------------------------------------#
def preprocessingCSV(anndata, delim='comma', transform='log', cellRatio=0.99, geneRatio=0.99, geneCriteria='variance', geneSelectnum=2000, transpose=False, tabuCol=''):
    '''
    preprocessing CSV files:
    transform='log' or 'None'
    '''

    tabuColList = []
    tmplist = tabuCol.split(",")
    for item in tmplist:
        tabuColList.append(item)
    index_col = []
    
    # print('---------------anndata.X.A.T----------------',anndata.X.A.T.shape[1])
    for i in range(0,anndata.X.A.T.shape[1]):
        index_col.append(i)
        
    adata_gene_list = anndata.var.index.tolist()                                #add
    adata_barcode_list = anndata.obs.index.tolist()                             #add
    df = pd.DataFrame()
    df = anndata.X.A.T
    print('Data loaded, start filtering...')
    
    if transpose == True:
        df = df.T
    # df.columns.name = index_col
    #df0 = pd.DataFrame(df,columns=index_col)
    df0 = pd.DataFrame(df,columns=adata_barcode_list,index=adata_gene_list)                       #add pd_index
    #print('df0 is !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', df0)                        #add pd_index
    df = df0
    df1 = df[df.astype('bool').mean(axis=1) >= (1-geneRatio)]
    print('After preprocessing, {} genes remaining'.format(df1.shape[0]))
    
    criteriaGene = df1.astype('bool').mean(axis=0) >= (1-cellRatio)
    df2 = df1[df1.columns[criteriaGene]]########
    print('After preprocessing, {} cells have {} nonzero'.format(
        df2.shape[1], geneRatio))
    
    criteriaSelectGene = df2.var(axis=1).sort_values()[-geneSelectnum:]
    df3 = df2.loc[criteriaSelectGene.index]
    if transform == 'log':
        df3 = df3.transform(lambda x: np.log(x + 1))
    #df3 is Use_expression.csv
    # df3.to_csv(csvFilename)
    #print('---------------------Use_expression---------------------------')

    return df3

def generate_coords_use_exp(anndata, sample, metadata_path):

    coords_list = [list(t) for t in zip(anndata.obs["array_row"].tolist(), anndata.obs["array_col"].tolist())]
    coords = np.array(coords_list)

    use_expression = preprocessingCSV(anndata, 'comma', None, 1.00, 0.99, 'variance', 2000, False,'')
    # print("Preprocessing Done. Total Running Time: %s seconds" %
    #       (time.time() - start_time))
    
    #generate label test
    metadata_pd = pd.read_csv(metadata_path+'/'+sample+'_humanBrain_metaData.csv',index_col=0)
    metadata_modify_barcode_pd = metadata_pd.loc[anndata.obs.index]
    
    if sample == '2-5' or sample == '2-8' or sample == '18-64' or sample == 'T4857':
        print('The sample is in the 4_samples!')
        ground_truth_label_np = np.zeros((metadata_modify_barcode_pd.shape[0],)).astype(int)
        ground_truth_init_np = np.array(metadata_modify_barcode_pd['benmarklabel'])
        
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
        print('ground_truth_label_np shape is', ground_truth_label_np.shape)
    
    if sample == '151507' or sample == '151508' or sample == '151509' or sample == '151510' or sample == '151669' or sample == '151670' or sample == '151671' or sample == '151672' or sample == '151673' or sample == '151674' or sample == '151675' or sample == '151676':
        print('The sample is in the 12_samples!')
        ground_truth_label_np = np.zeros((metadata_modify_barcode_pd.shape[0],)).astype(int)
        ground_truth_init_np = np.array(metadata_modify_barcode_pd['benmarklabel'])
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
        print('ground_truth_label_np shape is', ground_truth_label_np.shape)    

    #numpy transform
    use_expression_np = use_expression.to_numpy()
    use_expression_np = use_expression_np.T

    return coords, use_expression_np, ground_truth_label_np

def generate_coords_use_exp_st(anndata, select_gene_num):

    coords_list = [list(t) for t in zip(anndata.obs["array_row"].tolist(), anndata.obs["array_col"].tolist())]
    coords = np.array(coords_list)

    use_expression = preprocessingCSV(anndata, 'comma', None, 1.00, 0.99, 'variance', select_gene_num, False,'')
    # print("Preprocessing Done. Total Running Time: %s seconds" %
    #       (time.time() - start_time))

    #numpy transform
    use_expression_np = use_expression.to_numpy()
    use_expression_np = use_expression_np.T

    return coords, use_expression_np

#-------------------------------------HE data preprocess tile------------------------------------------#
def image_data_prepare_tile_st(img_3d, crop_radius, spot_row_in_fullres, spot_col_in_fullres, patch_target_size, adata_loc, tile_path): 

    #generate 3d image data
    init_img_3d = np.ones(shape=(len(spot_row_in_fullres),patch_target_size,patch_target_size,3), dtype=np.uint8)
    img_pillow = Image.fromarray(img_3d)

    for crop_index in range(len(spot_row_in_fullres)):
        adata_loc_current = adata_loc[crop_index]
        imagerow = spot_row_in_fullres[crop_index]
        imagecol = spot_col_in_fullres[crop_index]     
        imagerow_down  = imagerow - crop_radius
        imagerow_up    = imagerow + crop_radius
        imagecol_left  = imagecol - crop_radius
        imagecol_right = imagecol + crop_radius
        tile           = img_pillow.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
        #print('tile', tile)
        tile.thumbnail((patch_target_size, patch_target_size), Image.ANTIALIAS)
        tile.resize((patch_target_size, patch_target_size))        
        cropped_3d_current = np.asarray(tile)
        #print('cropped_3d_current is', cropped_3d_current)
        #print('cropped_3d_current shape is', cropped_3d_current.shape)
        #print('cropped_3d_current max is', np.max(cropped_3d_current))
        init_img_3d[crop_index,:,:,:] = cropped_3d_current
        #tile_name = str(adata_loc_current) + '_' + str(crop_radius*2)
        #tile.save(tile_path+tile_name+'.jpeg', "JPEG")       
    print('init_img_3d before transpose shape is', init_img_3d.shape)
    init_img_3d = init_img_3d.transpose((0,3,1,2))
    print('init_img_3d after transpose shape is', init_img_3d.shape)
    cropped_img_3d_nor = init_img_3d/255
    print('cropped_img_3d_nor max is ', np.max(cropped_img_3d_nor))
    return cropped_img_3d_nor