import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import scanpy as sc
from PIL import Image
import json

def load_data(h5_path, spatial_path, scale_factor_path, transform_opt):

    # Read in gene expression and spatial location
    adata = sc.read_10x_h5(h5_path)
    spatial_all = pd.read_csv(spatial_path, sep=",", header=None, na_filter=False, index_col=0)
    spatial = spatial_all[spatial_all[1] == 1]
    spatial = spatial.sort_values(by=0)
    assert all(adata.obs.index == spatial.index)
    adata.obs["in_tissue"] = spatial[1]
    adata.obs["array_row"] = spatial[2]
    adata.obs["array_col"] = spatial[3]
    adata.obs["pxl_col_in_fullres"] = spatial[4]
    adata.obs["pxl_row_in_fullres"] = spatial[5]
    adata.obs.index.name = 'barcode'
    adata.var_names_make_unique()

    # Read scale_factor_file
    with open(scale_factor_path) as fp_scaler:
        scaler = json.load(fp_scaler)
    adata.uns["spot_diameter_fullres"] = scaler["spot_diameter_fullres"]
    adata.uns["tissue_hires_scalef"] = scaler["tissue_hires_scalef"]
    adata.uns["fiducial_diameter_fullres"] = scaler["fiducial_diameter_fullres"]
    adata.uns["tissue_lowres_scalef"] = scaler["tissue_lowres_scalef"]
    
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

def generate_coords_use_exp_no_label(anndata):

    coords_list = [list(t) for t in zip(anndata.obs["array_row"].tolist(), anndata.obs["array_col"].tolist())]
    coords = np.array(coords_list)

    use_expression = preprocessingCSV(anndata, 'comma', None, 1.00, 0.99, 'variance', 2000, False,'')
    # print("Preprocessing Done. Total Running Time: %s seconds" %
    #       (time.time() - start_time))

    #numpy transform
    use_expression_np = use_expression.to_numpy()
    use_expression_np = use_expression_np.T

    return coords, use_expression_np

def load_chicken_heart_data(h5_path, spatial_path, scale_factor_path, transform_opt):
    # Read in gene expression and spatial location
    adata = sc.read_10x_h5(h5_path)
    spatial_all = pd.read_csv(spatial_path, sep=",", header=None, na_filter=False, index_col=0)
    spatial = spatial_all[spatial_all[1] == 1]
    spatial = spatial.sort_values(by=0)
    assert all(adata.obs.index == spatial.index)
    adata.obs["in_tissue"] = spatial[1]
    adata.obs["array_row"] = spatial[2]
    adata.obs["array_col"] = spatial[3]
    adata.obs["pxl_col_in_fullres"] = spatial[4]
    adata.obs["pxl_row_in_fullres"] = spatial[5]
    adata.obs.index.name = 'barcode'
    adata.var_names_make_unique()
    # Read scale_factor_file
    with open(scale_factor_path) as fp_scaler:
        scaler = json.load(fp_scaler)
    adata.uns["fiducial_diameter_fullres"] = scaler["oligo"][0]['dia']

    # transform optional
    if transform_opt == 'log':
        sc.pp.log1p(adata)
    elif transform_opt == 'logcpm':
        sc.pp.normalize_total(adata,target_sum=1e4)
        sc.pp.log1p(adata)
    elif transform_opt == 'None':
        transform_opt = 'raw'
    else:
        print('transform optional is log or logcpm or None')
    print('Original adata shape is', adata.X.A.shape)   #test
    
    return adata

def generate_coords_sc_chicken_heart(anndata, geneSelectnum):

    coords_list = [list(t) for t in zip(anndata.obs["array_row"].tolist(), anndata.obs["array_col"].tolist())]
    coords = np.array(coords_list)

    use_expression = preprocessingCSV(anndata, 'comma', None, 1.00, 0.99, 'variance', geneSelectnum, False,'')
    # print("Preprocessing Done. Total Running Time: %s seconds" %
    #       (time.time() - start_time))
    
    #numpy transform
    use_expression_np = use_expression.to_numpy()
    use_expression_np = use_expression_np.T
    print('use_expression_np max is',np.max(use_expression_np))
    print('use_expression_np min is',np.min(use_expression_np))

    return coords, use_expression_np

def normalization_min_max(inputdata):
    _range = np.max(inputdata) - np.min(inputdata)
    return (inputdata - np.min(inputdata)) / _range

#rna data preprocess
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

#image data preprocess tile for 3d
def image_data_prepare_tile(img_3d, crop_radius, spot_row_in_fullres, spot_col_in_fullres, patch_target_size, adata_barcode, tile_path):

    #generate 3d image data
    init_img_3d = np.ones(shape=(len(spot_row_in_fullres),patch_target_size,patch_target_size,3), dtype=np.uint8)
    img_pillow = Image.fromarray(img_3d)

    for crop_index in range(len(spot_row_in_fullres)):
        barcode = adata_barcode[crop_index]
        imagerow = spot_row_in_fullres[crop_index]
        imagecol = spot_col_in_fullres[crop_index]     
        imagerow_down  = imagerow - crop_radius
        imagerow_up    = imagerow + crop_radius
        imagecol_left  = imagecol - crop_radius
        imagecol_right = imagecol + crop_radius
        tile           = img_pillow.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
        tile.thumbnail((patch_target_size, patch_target_size), Image.ANTIALIAS)
        tile.resize((patch_target_size, patch_target_size))        
        cropped_3d_current = np.asarray(tile)
        init_img_3d[crop_index,:,:,:] = cropped_3d_current

    init_img_3d_used = np.ones(shape=(len(spot_row_in_fullres),crop_radius*2*crop_radius*2*3), dtype=np.uint8)
    for num in range(len(spot_row_in_fullres)):
        cropped_3d_used_current = init_img_3d[num,:,:,:].reshape(1,-1)
        init_img_3d_used[num,:] = cropped_3d_used_current

    #no nor cropped data
    cropped_img_3d_used = init_img_3d_used
    #nor cropped data
    cropped_img_3d_used_nor = cropped_img_3d_used/255
    
    print('cropped_img_3d_used max is ', np.max(cropped_img_3d_used))
    print('cropped_img_3d_used_nor max is ', np.max(cropped_img_3d_used_nor))
    return cropped_img_3d_used, cropped_img_3d_used_nor

#image data preprocess for gray
def image_data_prepare_for_original_gray(img_gray, radius, spot_row_in_fullres, spot_col_in_fullres):
    #generate image data
    init_img_gray = np.ones(shape=(len(spot_row_in_fullres),radius*2*radius*2), dtype=np.uint8)
    for crop_index in range(len(spot_row_in_fullres)):
        cropped_gray_img_current = img_gray[(spot_row_in_fullres[crop_index]-radius):(spot_row_in_fullres[crop_index]+radius),(spot_col_in_fullres[crop_index]-radius):(spot_col_in_fullres[crop_index]+radius)].reshape(1,-1)
        init_img_gray[crop_index,:] = cropped_gray_img_current
    cropped_gray_img = init_img_gray
    cropped_gray_img_nor = init_img_gray/255
    print('cropped_gray_img max is ',np.max(cropped_gray_img))
    print('cropped_gray_img_nor max is ',np.max(cropped_gray_img_nor))
    return cropped_gray_img, cropped_gray_img_nor

#image data preprocess for 3d
def image_data_prepare_for_original_3d(img_3d, radius, spot_row_in_fullres, spot_col_in_fullres):
    #generate 3d HE data for resnet50
    init_img_3d = np.ones(shape=(len(spot_row_in_fullres),radius*2*radius*2*3), dtype=np.uint8)
    for crop_index in range(len(spot_row_in_fullres)):
        cropped_img_3d_current = img_3d[(spot_row_in_fullres[crop_index]-radius):(spot_row_in_fullres[crop_index]+radius),(spot_col_in_fullres[crop_index]-radius):(spot_col_in_fullres[crop_index]+radius),:].reshape(1,-1)
        init_img_3d[crop_index,:] = cropped_img_3d_current
    cropped_img_3d = init_img_3d
    cropped_img_3d_nor = init_img_3d/255
    print('cropped_img_3d max is ',np.max(cropped_img_3d))
    print('cropped_img_3d_nor max is ',np.max(cropped_img_3d_nor))
    return cropped_img_3d, cropped_img_3d_nor


#kmeans result
def Embedding_Kmeans_Result(embedding, sample, adata, metadata_path):
    #set embedding
    emb_evaluation = embedding
    n_clusters_num = 7
    if sample=='151669' or sample=='151670' or sample=='151671' or sample=='151672':
        n_clusters_num = 5
    if sample=='2-8':
        n_clusters_num = 6
        
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
        print('ground_truth_label_np shape is', ground_truth_label_np.shape)
        ari = adjusted_rand_score(kmeans_label_pre_filter , ground_truth_label_np)
        print('ari',ari)
    return ari

def Embedding_Kmeans_Result_Chicken_Heart(embedding, sample, adata, metadata_path):
    #set embedding
    emb_evaluation = embedding
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
    #print('ground_truth_label_np', ground_truth_label_np)
    print('ground_truth_label_np unique num is',len(np.unique(ground_truth_label_np)))
    print('ground_truth_label_np shape is', ground_truth_label_np.shape)

    #kmeans
    X = emb_evaluation
    kmeans = KMeans(n_clusters=n_clusters_num, random_state=0).fit(X)
    kmeans_label = kmeans.labels_
    #print('kmeans_label', kmeans_label)
    print('kmeans_label unique num is',len(np.unique(kmeans_label)))
    print('kmeans_label shape is', kmeans_label.shape)
    ari = adjusted_rand_score(kmeans_label , ground_truth_label_np)
    print('ari',ari)
    
    return ari