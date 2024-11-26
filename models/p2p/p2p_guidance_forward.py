import os
import numpy as np
import torch

from models.p2p.attention_control import register_attention_control,set_note,clear_note,set_invert_note,clear_invert_note,get_invert_note
from utils.utils import init_latent
import pdb
from collections import defaultdict
import torch.nn.functional as nnf
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
    

torch.autograd.set_detect_anomaly(True)


def dilate(image, kernel_size, stride=1, padding=0):
    """
    Perform dilation on a binary image using a square kernel.
    """
    # Ensure the image is binary
    assert image.max() <= 1 and image.min() >= 0
    
    # Get the maximum value in each neighborhood
    dilated_image = nnf.max_pool2d(image, kernel_size, stride, padding)
    
    return dilated_image


def p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False,
                                edit_stage=True, prox=None, quantile=0.7,
                   image_enc=None, recon_lr=0.1, recon_t=400,
                   inversion_guidance=False, x_stars=None, i=0,
                   dilate_mask=0,latentsuam=None,):

    if low_resource: 

        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1].unsqueeze(0))["sample"]

    else:
        latents_input = torch.cat([latents] * 2) 


        try:
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"] 
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            noise_pred_uam = model.unet(latentsuam, t, encoder_hidden_states=context[0].unsqueeze(0))["sample"]
        except:

            pdb.set_trace()


    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    recon_t_end=0
    recon_t_begin = recon_t
    
    quantile_list=np.linspace(0, quantile, 50)
    if ( (t < recon_t_begin) and ( t > recon_t_end)):
        
        prev_t = t - model.scheduler.config.num_train_timesteps  // model.scheduler.num_inference_steps
        alpha_prod_t = model.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_t] if prev_t > 0 else model.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (latents - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
        pred_x0_uam = (latentsuam - beta_prod_t**0.5 * noise_pred_uam) / alpha_prod_t**0.5
        

        x0_delta = (pred_x0[0]-pred_x0[1])

        threshold = x0_delta.abs().quantile(quantile_list[i])

        x0_delta -= x0_delta.clamp(-threshold, threshold) 


        mask_edit = (x0_delta.abs() > threshold).float()

        radius = int(dilate_mask)
        mask_edit = dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)    
        recon_mask = 1 - mask_edit

        
        pred_x0[1] = pred_x0[1] -  (pred_x0[1] - pred_x0_uam) * recon_mask
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * noise_pred
        latents = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir


        pred_dir_uam = (1 - alpha_prod_t_prev)**0.5 * noise_pred_uam
        latentsuam = alpha_prod_t_prev**0.5 * pred_x0_uam + pred_dir_uam
                        
    else:  
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        latentsuam = model.scheduler.step(noise_pred_uam, t, latentsuam)["prev_sample"]



    latents = controller.step_callback(latents)
    return latents,latentsuam



@torch.no_grad()
def p2p_guidance_forward(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None,

    low_resource=False,
    edit_stage=True,
    prox=None,
    quantile=0.7,
    image_enc=None,
    recon_lr=0.1,
    recon_t=400, 
    inversion_guidance=False,
    x_stars=None,
    dilate_mask=None,
    start_code_uam=None,
):
    batch_size = len(prompt)

    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)


    latents_list=[latents]


    prox="l0"

    inversion_guidance=True
    print("w/ mask")

           
    print(f"recon_t:{recon_t}, quantile:{quantile}")
    
    latentsuam=start_code_uam
    latents_list_uam=[latentsuam]

    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])

        latents,latentsuam = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale,x_stars=x_stars,
                                            edit_stage=edit_stage, prox=prox, quantile=quantile,
                                           image_enc=image_enc, recon_lr=recon_lr, recon_t=recon_t,
                                           inversion_guidance=inversion_guidance, i=i,dilate_mask=dilate_mask,latentsuam=latentsuam,
                                              )
        


    return latents, latent,latents_list,latents_list_uam

@torch.no_grad()
def p2p_guidance_forward_single_branch(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([torch.cat([uncond_embeddings[i],uncond_embeddings_[1:]]), text_embeddings])
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    return latents, latent


def direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True,edit_stage=True, prox=None, quantile=0.7,
                   image_enc=None, recon_lr=0.1, recon_t=0,
                   inversion_guidance=False, x_stars=None, i=0,
                   dilate_mask=0,latentsuam=None,return_intermediates=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        
        noise_pred_uam = model.unet(latentsuam, t, encoder_hidden_states=context[0].unsqueeze(0))["sample"]
 

        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)



    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    recon_t_end=0
    recon_t_begin = recon_t   
    quantile_list=np.linspace(0, quantile, 50)
    if ( (t < recon_t_begin) and ( t > recon_t_end)):
        
        print(f"a quantile:{quantile}:{quantile_list[i]}, recon_t_end:{recon_t_end}")
        prev_t = t - model.scheduler.config.num_train_timesteps  // model.scheduler.num_inference_steps
        alpha_prod_t = model.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_t] if prev_t > 0 else model.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (latents - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
        pred_x0_uam = (latentsuam - beta_prod_t**0.5 * noise_pred_uam) / alpha_prod_t**0.5
        

        x0_delta = (pred_x0[0]-pred_x0[1])
        

        threshold = x0_delta.abs().quantile(quantile_list[i])

        x0_delta -= x0_delta.clamp(-threshold, threshold) 


        mask_edit = (x0_delta.abs() > threshold).float()

        radius = int(dilate_mask)
        mask_edit = dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)    
        recon_mask = 1 - mask_edit

        pred_x0[1] = pred_x0[1] - (pred_x0[1] - pred_x0_uam) * recon_mask
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * noise_pred
        latents = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir


        pred_dir_uam = (1 - alpha_prod_t_prev)**0.5 * noise_pred_uam
        latentsuam = alpha_prod_t_prev**0.5 * pred_x0_uam + pred_dir_uam
                        
    else:  
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]



    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:])) 
    latents = controller.step_callback(latents)
    return latents,latentsuam


def direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]+noise_loss[1:]))
    latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def direct_inversion_p2p_guidance_forward(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    noise_loss_list = None,
    add_offset=True,motivation_latents=None,
    edit_stage=True,
    prox=None,
    quantile=0.7,
    image_enc=None,
    recon_lr=0.1,
    recon_t=0,
    inversion_guidance=False,
    x_stars=None,
    dilate_mask=None,
    start_code_uam=None,
    return_intermediates=False
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    latents_list=[latents]



    prox="l0"

    inversion_guidance=True
    print("w/ mask")

        
    latentsuam=start_code_uam
    latents_list_uam=[latentsuam]
    for i, t in enumerate(model.scheduler.timesteps):
        
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents,latentsuam = direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss_list[i],low_resource=False,add_offset=add_offset,
                                                               x_stars=x_stars,
                                            edit_stage=edit_stage, prox=prox, quantile=quantile,
                                           image_enc=image_enc, recon_lr=recon_lr, recon_t=recon_t,
                                           inversion_guidance=inversion_guidance, i=i,dilate_mask=dilate_mask,
                                            latentsuam=latentsuam,
                    
                                                               )

        latents_list.append(latents)
        latents_list_uam.append(latentsuam)

    return latents, latent,latents_list,latents_list_uam

@torch.no_grad()
def direct_inversion_p2p_guidance_forward_add_target(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    noise_loss_list = None,
    add_offset=True
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss_list[i],low_resource=False,add_offset=add_offset)
        
    return latents, latent
