import os
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from CL_data_preprocess import *
from CL_model import *
from util_function import loss_function
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from losses import SupConLoss
import torch.utils.data as DataLoad
import pandas as pd
from sklearn.preprocessing import normalize
import torchvision.models as models

def Generate_CL_Embedding(torch_dataset, gae_hidden1_dim, gae_hidden2_dim, gcn_encoder_dropout, gcn_decoder_dropout, k_graph, cl_emb_dim, temperature_cl, image_arch, batch_size, model_lr, model_epoch, w_rna_image, w_rna_gae, adata, learning_name, output_path, device):
    '''
    Contrastive Learning Embedding
    Return:
        Embedding
    '''
    
    #DataLoader by batch_size
    dataset_loader = DataLoad.DataLoader(dataset = torch_dataset, batch_size = batch_size, shuffle = False, drop_last=False)
    dataset_loader_eval = DataLoad.DataLoader(dataset = torch_dataset, batch_size = 1, shuffle = False, drop_last=False)
    
    #DataLoader by dataset
    rna_np = torch_dataset.tensors[0].numpy()
    coords_np = torch_dataset.tensors[1].numpy()
    input_feat_dim_rna = rna_np.shape[1]
    device_used = device
    
    #model setting
    CL_lr = model_lr
    CL_epoch = model_epoch
    
    #load CL model
    model = CL_Model(input_feat_dim_rna, gae_hidden1_dim, gae_hidden2_dim, gcn_encoder_dropout, gcn_decoder_dropout, cl_emb_dim, models.__dict__[image_arch]())
    #print(model)
    
    optimizer = optim.Adam(model.parameters(), lr=CL_lr)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=500, verbose=True)
    device=torch.device(device)
    model = model.to(device)
    
    ##---------------model save test start------------------##
    #ptfileStart = 'GCN_RESNET_EMtrainingStart.pt'
    #stateStart = {
    #    'state_dict': model.state_dict(),
    #    'optimizer': optimizer.state_dict(),
    #}
    #torch.save(stateStart, output_path+'/model_save/'+learning_name+'_'+ptfileStart)

    #spatial data preprocess
    train_rna_features = rna_np
    train_coords_np = coords_np
    train_adj = coords_to_adj(train_rna_features,train_coords_np,k_graph)

    #adj preprocess
    adj_norm, adj_label, pos_weight, norm = graph_processing(train_adj)
    
    #transform to tensor
    train_rna_features = torch.FloatTensor(train_rna_features)
    n_nodes = train_rna_features.shape[0]
    
    #to device
    adj_label = adj_label.to(device)
    adj_norm = adj_norm.to(device)
    train_rna_features = train_rna_features.to(device)

    #-----------save loss-------#
    gae_loss_list = []
    cl_rna_rna_loss_list = []
    cl_rna_image_loss_list = []
    final_loss_list = []

    for epoch in tqdm(range(CL_epoch)):
        epoch_current = epoch + 1
        print('epoch_current',epoch_current)
        batch_num=0
        for batch_rna, batch_coords, batch_image, batch_order in dataset_loader:
            batch_num = batch_num + 1
            print('batch_num',batch_num)

            #current batch_size
            batch_order_np = batch_order.numpy()
            batch_size_current = batch_order_np.shape[0]
            
            #image batch
            #if batch_size_current == train_rna_features.shape[0] and w_rna_image == 0:
            #    order_batch = np.arange(0,20,1)
            #    batch_image = batch_image[order_batch]
            if w_rna_image == 0:                   #modified   #modified add
                #order_batch = np.arange(0,20,1)               #modified add
                order_batch = np.arange(0,1,1)                 #modified add
                batch_image = batch_image[order_batch]         #modified add

            batch_image = batch_image.to(device)
                        
            #model train
            model.train()
            optimizer.zero_grad()
            
            rna_emb, hidden2_rna, logvar_rna, image_emb, image_encoder_emb, logvar_image = model(train_rna_features, adj_norm, batch_image)
            gae_preds_rna = model.decoder(hidden2_rna)
            
            #model gae loss 
            gae_loss = loss_function(preds=gae_preds_rna, labels=adj_label, mu=hidden2_rna, logvar=logvar_rna, n_nodes=n_nodes, norm=norm, pos_weight=pos_weight)

            ##model cl loss 
            if w_rna_image == 0:                   #modified add
                batch_order_np = order_batch       #modified add
            rna_cl_train = rna_emb[batch_order_np]
            image_cl_train = image_emb

            #rna_image_cl_loss
            rna_cl_train_l2_norm = F.normalize(rna_cl_train, dim=1)
            if batch_size_current == train_rna_features.shape[0] and w_rna_image == 0:
                rna_cl_train_l2_norm = rna_cl_train_l2_norm[order_batch]
                image_cl_train_l2_norm = F.normalize(image_cl_train, dim=1)
                features_cl_rna_image = torch.cat([rna_cl_train_l2_norm.unsqueeze(1), image_cl_train_l2_norm.unsqueeze(1)], dim=1)
                criterion_cl_rna_image = SupConLoss(temperature=temperature_cl,base_temperature=temperature_cl)
                simclr_loss_rna_image = criterion_cl_rna_image(features_cl_rna_image)
            else:
                image_cl_train_l2_norm = F.normalize(image_cl_train, dim=1)
                features_cl_rna_image = torch.cat([rna_cl_train_l2_norm.unsqueeze(1), image_cl_train_l2_norm.unsqueeze(1)], dim=1)
                criterion_cl_rna_image = SupConLoss(temperature=temperature_cl,base_temperature=temperature_cl)
                simclr_loss_rna_image = criterion_cl_rna_image(features_cl_rna_image)

            #final loss 
            loss = simclr_loss_rna_image*w_rna_image + gae_loss*w_rna_gae

            print('simclr_loss_rna_image is',simclr_loss_rna_image)
            print('gae_loss is',gae_loss)                          
            print('loss',loss)                                     

            loss.backward()
            cur_loss = loss.item()
            optimizer.step()
            print("Epoch "+str(epoch_current)+': loss--->',cur_loss)
            #scheduler.step(loss)

            #----------save loss--------#
            gae_loss_no_tensor = transform_tensor_to_numpy(gae_loss,device_used)
            cl_image_rna_loss_no_tensor = transform_tensor_to_numpy(simclr_loss_rna_image,device_used)
            final_loss_no_tensor = transform_tensor_to_numpy(loss,device_used)
            
            gae_loss_list.append(gae_loss_no_tensor)
            cl_rna_image_loss_list.append(cl_image_rna_loss_no_tensor)
            final_loss_list.append(final_loss_no_tensor)

        validate_model_epoch = 5
        #if epoch_current % 50 == 0 or epoch_current == 1:
        #if epoch_current % 30 == 0 or epoch_current == 1:
        if epoch_current % validate_model_epoch == 0:
            #valid model
            feature_len = rna_np.shape[0]
            feature_dim_emb1_cl2 = image_emb.shape[1]
            feature_dim_emb2_cl2 = image_encoder_emb.shape[1]
            feature_dim_emb1_cl1 = rna_emb.shape[1]
            feature_dim_emb2_cl1 = hidden2_rna.shape[1]
            valid_model(model, dataset_loader_eval, rna_np, coords_np, k_graph, device_used, feature_len, feature_dim_emb1_cl1, feature_dim_emb2_cl1, feature_dim_emb1_cl2, feature_dim_emb2_cl2, adata, train_rna_features, adj_norm, output_path, learning_name, epoch_current)
            
            #---------------model save test end------------------##
            #ptfileEnd = 'GCN_RESNET_EMtrainingEnd.pt'
            #stateEnd = {
            #    'state_dict': model.state_dict(),
            #    'optimizer': optimizer.state_dict(),
            #}
            #torch.save(stateEnd, output_path+'/model_save/'+learning_name+'_epoch'+str(epoch_current)+'_'+ptfileEnd)
    tqdm.write("Optimization Finished!")

    #------------save loss------------#
    loss_path = output_path+'/loss/'
    gae_loss_list_np = np.array(gae_loss_list)
    cl_rna_image_loss_list_np = np.array(cl_rna_image_loss_list)
    final_loss_list_np = np.array(final_loss_list)
    gae_loss_list_pd = pd.DataFrame(gae_loss_list_np)
    cl_rna_image_loss_list_pd = pd.DataFrame(cl_rna_image_loss_list_np)
    final_loss_list_pd = pd.DataFrame(final_loss_list_np)
    gae_loss_list_pd.to_csv(loss_path+'/gae_loss_epoch'+str(epoch_current)+'_'+learning_name+'.csv')
    cl_rna_image_loss_list_pd.to_csv(loss_path+'/cl_rna_image_loss_epoch'+str(epoch_current)+'_'+learning_name+'.csv')
    final_loss_list_pd.to_csv(loss_path+'/final_loss_epoch'+str(epoch_current)+'_'+learning_name+'.csv')

    #return print('CL finished!')
    return validate_model_epoch


def valid_model(model, dataset_loader_eval, rna_np, coords_np, k_graph, device_used, feature_len, feature_dim_emb1_cl1, feature_dim_emb2_cl1, feature_dim_emb1_cl2, feature_dim_emb2_cl2, adata, train_rna_features, adj_norm, output_path, learning_name, epoch_current):
    
    with torch.no_grad():
        model.eval()
        device=torch.device(device_used)
        model = model.to(device)

        #image data preprocess
        source_data_emb1_cl2 = []
        source_data_emb2_cl2 = []
        all_samples = 0
        source_data_emb1_cl2_np = np.zeros((feature_len, feature_dim_emb1_cl2))
        source_data_emb2_cl2_np = np.zeros((feature_len, feature_dim_emb2_cl2))

        #rna full data preprocess 
        features_full = rna_np
        coords_np_full = coords_np
        adj_full = coords_to_adj(features_full, coords_np_full, k_graph)
        adj_norm_full, adj_label_full, pos_weight_full, norm_full = graph_processing(adj_full)
        features_full = torch.FloatTensor(features_full)
        features_full = features_full.to(device)
        adj_norm_full = adj_norm_full.to(device)

        if device_used == 'cuda':
            assert (features_full.cpu().detach().numpy() == train_rna_features.cpu().detach().numpy()).all()
            assert (adj_norm_full.cpu().to_dense().detach().numpy() == adj_norm.cpu().to_dense().detach().numpy()).all()
        else:
            assert (features_full.detach().numpy() == train_rna_features.detach().numpy()).all()
            assert (adj_norm_full.to_dense().detach().numpy() == adj_norm.to_dense().detach().numpy()).all()

        #rna eval data preprocess 
        features_eval = rna_np[0:30,:]
        coords_np_eval = coords_np[0:30,:]
        adj_eval = coords_to_adj(features_eval, coords_np_eval, k_graph)
        adj_norm_eval, adj_label_eval, pos_weight_eval, norm_eval = graph_processing(adj_eval)
        features_eval = torch.FloatTensor(features_eval)
        features_eval = features_eval.to(device)
        adj_norm_eval = adj_norm_eval.to(device)

        #image data validation
        for batch_rna_eval, batch_coords_eval, batch_image_eval, batch_order_eval in dataset_loader_eval:
            batch_image_eval = batch_image_eval.to(device)
            model.eval()
            rna_emb_eval, hidden2_rna_eval, logvar_rna_eval, image_emb_eval, image_encoder_emb_eval, logvar_image_eval = model(features_eval, adj_norm_eval, batch_image_eval) 
            if device_used == 'cuda':
                source_data_emb1_cl2_np[all_samples:all_samples + image_emb_eval.shape[0], :] = image_emb_eval.cpu().detach().numpy() 
                source_data_emb2_cl2_np[all_samples:all_samples + image_encoder_emb_eval.shape[0], :] = image_encoder_emb_eval.cpu().detach().numpy()
            else:
                source_data_emb1_cl2_np[all_samples:all_samples + image_emb_eval.shape[0], :] = image_emb_eval.detach().numpy()
                source_data_emb2_cl2_np[all_samples:all_samples + image_encoder_emb_eval.shape[0], :] = image_encoder_emb_eval.detach().numpy()
            all_samples += image_encoder_emb_eval.shape[0]
        source_data_emb1_cl2.extend(source_data_emb1_cl2_np[:all_samples])
        source_data_emb1_cl2 = np.array(source_data_emb1_cl2).astype('float32')
        source_data_emb2_cl2.extend(source_data_emb2_cl2_np[:all_samples])
        source_data_emb2_cl2 = np.array(source_data_emb2_cl2).astype('float32')
        source_data_emb1_cl2_norm = normalize(source_data_emb1_cl2,norm='l2')

        #rna data validation
        model.eval()
        rna_emb_full, hidden2_rna_full, logvar_rna_full, image_emb_full, image_encoder_emb_full, logvar_image_full = model(features_full, adj_norm_full, batch_image_eval)
        rna_emb_full1, hidden2_rna_full1, logvar_rna_full1, image_emb_full1, image_encoder_emb_full1, logvar_image_full1 = model(features_full, adj_norm_full, batch_image_eval)
        if device_used == 'cuda':
            assert (rna_emb_full.cpu().detach().numpy() == rna_emb_full1.cpu().detach().numpy()).all()
            assert (image_emb_full.cpu().detach().numpy() == image_emb_full1.cpu().detach().numpy()).all()
            source_data_emb1_cl1 = rna_emb_full.cpu().detach().numpy()
            source_data_emb2_cl1 = hidden2_rna_full.cpu().detach().numpy()
            source_data_emb1_cl1_norm = normalize(source_data_emb1_cl1,norm='l2')
        else:
            assert (rna_emb_full.detach().numpy() == rna_emb_full1.detach().numpy()).all()
            assert (image_emb_full.detach().numpy() == image_emb_full1.detach().numpy()).all()
            source_data_emb1_cl1 = rna_emb_full.detach().numpy()
            source_data_emb2_cl1 = hidden2_rna_full.detach().numpy()
            source_data_emb1_cl1_norm = normalize(source_data_emb1_cl1,norm='l2')

    #save embedding
    embedding_path = output_path+'/embedding/'
    emb1_cl1_list = []
    emb2_cl1_list = []
    emb1_cl2_list = []
    emb2_cl2_list = []
    
    for emb_num in range(feature_dim_emb1_cl1):       
        emb1_cl1_list.append('embedding'+str(emb_num)) 
    for emb_num in range(feature_dim_emb2_cl1):       
        emb2_cl1_list.append('embedding'+str(emb_num)) 
    for emb_num in range(feature_dim_emb1_cl2):     
        emb1_cl2_list.append('embedding'+str(emb_num)) 
    for emb_num in range(feature_dim_emb2_cl2):
        emb2_cl2_list.append('embedding'+str(emb_num)) 

    embedding1_cl1_df = pd.DataFrame(source_data_emb1_cl1, columns=emb1_cl1_list, index = adata.obs.index)
    embedding1_cl1_df.to_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_current)+'_emb1_cl1.csv')
    embedding1_cl2_df = pd.DataFrame(source_data_emb1_cl2, columns=emb1_cl2_list, index = adata.obs.index)
    embedding1_cl2_df.to_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_current)+'_emb1_cl2.csv')

    embedding2_cl1_df = pd.DataFrame(source_data_emb2_cl1, columns=emb2_cl1_list, index = adata.obs.index)
    embedding2_cl1_df.to_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_current)+'_emb2_cl1.csv')
    embedding2_cl2_df = pd.DataFrame(source_data_emb2_cl2, columns=emb2_cl2_list, index = adata.obs.index)
    embedding2_cl2_df.to_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_current)+'_emb2_cl2.csv')

    embedding1_cl1_norm_df = pd.DataFrame(source_data_emb1_cl1_norm, columns=emb1_cl1_list, index = adata.obs.index)
    embedding1_cl1_norm_df.to_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_current)+'_emb1_norm_cl1.csv')
    embedding1_cl2_norm_df = pd.DataFrame(source_data_emb1_cl2_norm, columns=emb1_cl2_list, index = adata.obs.index)
    embedding1_cl2_norm_df.to_csv(embedding_path+'/'+learning_name+'_epoch'+str(epoch_current)+'_emb1_norm_cl2.csv')

    return print('Current epoch model validation finished!')
