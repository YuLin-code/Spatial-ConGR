import pandas as pd
import numpy as np
import scanpy as sc
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='image quality evaluation')
    parser.add_argument('-sample', type=str, nargs='+', default=['151507'], help='which sample to be used: 151507-101510,151669-151676,2-5,2-8,18-64,T4857.')
    parser.add_argument('-transform_opt', type=str, nargs='+', default=['raw'], help='the data normalization for raw rna input.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    #load setting
    args = parse_args()
    sample = args.sample[0]
    transform_opt = args.transform_opt[0]

    #os
    result_path = './Image_Quality_Evaluation/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    #file
    #orginal_pca_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_original_transformed_'+transform_opt+'_3D_pca_transformed_lowres.png'
    orginal_pca_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_original_pre_transformed_'+transform_opt+'_3D_pca_transformed_lowres.png'
    #cl_gcnres_pca_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_g1L512_g2L128_drE0.0_drD0.0_k8_'+transform_opt+'_nor01_embD128_temp0.1_resnet18_batch64_lr0.001_epoch5_r2i1.0_gae0.0_epoch5_emb2_cl1_3D_pca_transformed_lowres.png'
    #cl_gaeres_pca_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_g1L512_g2L128_drE0.0_drD0.0_k8_'+transform_opt+'_nor01_embD128_temp0.1_resnet18_batch64_lr0.001_epoch5_r2i1.0_gae100.0_epoch5_emb2_cl1_3D_pca_transformed_lowres.png'
    cl_gcnres_pca_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_ConGcR_3D_pca_transformed_lowres.png'
    cl_gaeres_pca_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_ConGaR_3D_pca_transformed_lowres.png'

    #orginal_tsne_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_original_transformed_'+transform_opt+'_3D_tsne_transformed_lowres.png'
    orginal_tsne_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_original_pre_transformed_'+transform_opt+'_3D_tsne_transformed_lowres.png'
    #cl_gcnres_tsne_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_g1L512_g2L128_drE0.0_drD0.0_k8_'+transform_opt+'_nor01_embD128_temp0.1_resnet18_batch64_lr0.001_epoch5_r2i1.0_gae0.0_epoch5_emb2_cl1_3D_tsne_transformed_lowres.png'
    #cl_gaeres_tsne_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_g1L512_g2L128_drE0.0_drD0.0_k8_'+transform_opt+'_nor01_embD128_temp0.1_resnet18_batch64_lr0.001_epoch5_r2i1.0_gae100.0_epoch5_emb2_cl1_3D_tsne_transformed_lowres.png'
    cl_gcnres_tsne_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_ConGcR_3D_tsne_transformed_lowres.png'
    cl_gaeres_tsne_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_ConGaR_3D_tsne_transformed_lowres.png'

    #orginal_umap_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_original_transformed_'+transform_opt+'_3D_umap_transformed_lowres.png'
    orginal_umap_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_original_pre_transformed_'+transform_opt+'_3D_umap_transformed_lowres.png'
    #cl_gcnres_umap_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_g1L512_g2L128_drE0.0_drD0.0_k8_'+transform_opt+'_nor01_embD128_temp0.1_resnet18_batch64_lr0.001_epoch5_r2i1.0_gae0.0_epoch5_emb2_cl1_3D_umap_transformed_lowres.png'
    #cl_gaeres_umap_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_g1L512_g2L128_drE0.0_drD0.0_k8_'+transform_opt+'_nor01_embD128_temp0.1_resnet18_batch64_lr0.001_epoch5_r2i1.0_gae100.0_epoch5_emb2_cl1_3D_umap_transformed_lowres.png'
    cl_gcnres_umap_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_ConGcR_3D_umap_transformed_lowres.png'
    cl_gaeres_umap_file = './Embedding_RGB_Image/'+sample+'/'+sample+'_ConGaR_3D_umap_transformed_lowres.png'

    #read image
    orginal_pca_img = cv2.imread(orginal_pca_file)
    cl_gcnres_pca_img = cv2.imread(cl_gcnres_pca_file)
    cl_gaeres_pca_img = cv2.imread(cl_gaeres_pca_file)

    orginal_tsne_img = cv2.imread(orginal_tsne_file)
    cl_gcnres_tsne_img = cv2.imread(cl_gcnres_tsne_file)
    cl_gaeres_tsne_img = cv2.imread(cl_gaeres_tsne_file)

    orginal_umap_img = cv2.imread(orginal_umap_file)
    cl_gcnres_umap_img = cv2.imread(cl_gcnres_umap_file)
    cl_gaeres_umap_img = cv2.imread(cl_gaeres_umap_file)

    #calculate metrics
    original_cl_gcn_pca_psnr = compare_psnr(orginal_pca_img, cl_gcnres_pca_img)
    original_cl_gcn_pca_ssim = compare_ssim(orginal_pca_img, cl_gcnres_pca_img, multichannel=True)
    original_cl_gcn_pca_mse = compare_mse(orginal_pca_img, cl_gcnres_pca_img)
    
    original_cl_gae_pca_psnr = compare_psnr(orginal_pca_img, cl_gaeres_pca_img)
    original_cl_gae_pca_ssim = compare_ssim(orginal_pca_img, cl_gaeres_pca_img, multichannel=True)
    original_cl_gae_pca_mse = compare_mse(orginal_pca_img, cl_gaeres_pca_img)
    
    
    original_cl_gcn_tsne_psnr = compare_psnr(orginal_tsne_img, cl_gcnres_tsne_img)
    original_cl_gcn_tsne_ssim = compare_ssim(orginal_tsne_img, cl_gcnres_tsne_img, multichannel=True)
    original_cl_gcn_tsne_mse = compare_mse(orginal_tsne_img, cl_gcnres_tsne_img)
    
    original_cl_gae_tsne_psnr = compare_psnr(orginal_tsne_img, cl_gaeres_tsne_img)
    original_cl_gae_tsne_ssim = compare_ssim(orginal_tsne_img, cl_gaeres_tsne_img, multichannel=True)
    original_cl_gae_tsne_mse = compare_mse(orginal_tsne_img, cl_gaeres_tsne_img)
    
    
    original_cl_gcn_umap_psnr = compare_psnr(orginal_umap_img, cl_gcnres_umap_img)
    original_cl_gcn_umap_ssim = compare_ssim(orginal_umap_img, cl_gcnres_umap_img, multichannel=True)
    original_cl_gcn_umap_mse = compare_mse(orginal_umap_img, cl_gcnres_umap_img)
    
    original_cl_gae_umap_psnr = compare_psnr(orginal_umap_img, cl_gaeres_umap_img)
    original_cl_gae_umap_ssim = compare_ssim(orginal_umap_img, cl_gaeres_umap_img, multichannel=True)
    original_cl_gae_umap_mse = compare_mse(orginal_umap_img, cl_gaeres_umap_img)
    
    pca_metric_summary_np = np.array([original_cl_gcn_pca_psnr,original_cl_gcn_pca_ssim,original_cl_gcn_pca_mse,original_cl_gae_pca_psnr,original_cl_gae_pca_ssim,original_cl_gae_pca_mse]).reshape(1,-1)
    tsne_metric_summary_np = np.array([original_cl_gcn_tsne_psnr,original_cl_gcn_tsne_ssim,original_cl_gcn_tsne_mse,original_cl_gae_tsne_psnr,original_cl_gae_tsne_ssim,original_cl_gae_tsne_mse]).reshape(1,-1)
    umap_metric_summary_np = np.array([original_cl_gcn_umap_psnr,original_cl_gcn_umap_ssim,original_cl_gcn_umap_mse,original_cl_gae_umap_psnr,original_cl_gae_umap_ssim,original_cl_gae_umap_mse]).reshape(1,-1)
    
    metric_summary_np = np.vstack((pca_metric_summary_np,tsne_metric_summary_np,umap_metric_summary_np))
    metric_summary_pd = pd.DataFrame(metric_summary_np,columns=['cl_gcn_psnr','cl_gcn_ssim','cl_gcn_mse','cl_gae_psnr','cl_gae_ssim','cl_gae_mse'],index=['pca','tsne','umap'])
    
    metric_summary_pd.to_csv(result_path+sample+'_transform_opt_'+transform_opt+'_image_quality_evaluation.csv')