"""
Util functions based on Diffuser framework.
"""


import torch
import numpy as np

from tqdm import tqdm
from PIL import Image
import pdb
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf


def dilate(image, kernel_size, stride=1, padding=0):
    """
    Perform dilation on a binary image using a square kernel.
    """
    # Ensure the image is binary
    assert image.max() <= 1 and image.min() >= 0
    
    # Get the maximum value in each neighborhood
    dilated_image = nnf.max_pool2d(image, kernel_size, stride, padding)
    
    return dilated_image



class MasaCtrlPipeline(StableDiffusionPipeline):

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        start_code_uam=None,
        return_intermediates=False,
        noise_loss_list=None,edit_stage=True, prox=None, quantile=0.7,
                   image_enc=None, recon_lr=0.1, recon_t=400,
                   inversion_guidance=False, x_stars=None, i=0,
                   dilate_mask=0,reverse_pred_x0_list_uam=None,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        
        latents_list = [latents]
        pred_x0_list = [latents]
        pred_x0_uam_list = [start_code_uam]
        mask_list=[]

        inversion_guidance=True
        print("w/ mask")
   
        

        latentsuam=start_code_uam
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:

                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])


            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
     
            noise_pred_uam = self.unet(latentsuam, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample



            recon_t_end=0
            recon_t_begin = recon_t
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
            
            quantile_list=np.linspace(0, quantile, 50)
            if (inversion_guidance and (t < recon_t_begin) and ( t > recon_t_end)):
                recon_lr=1 
                prev_t = t - self.scheduler.config.num_train_timesteps  // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t > 0 else self.scheduler.final_alpha_cumprod
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

                
                # tgt branch
                pred_x0[1] = pred_x0[1] - recon_lr * (pred_x0[1] - pred_x0_uam) * recon_mask
                # src branch
                pred_x0[0] = pred_x0[0] - recon_lr * (pred_x0[0] - pred_x0_uam) * mask_edit
                
                
                pred_dir = (1 - alpha_prod_t_prev)**0.5 * noise_pred
                latents = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir


                pred_dir_uam = (1 - alpha_prod_t_prev)**0.5 * noise_pred_uam
                latentsuam = alpha_prod_t_prev**0.5 * pred_x0_uam + pred_dir_uam
                             
                
            else:
                latents, pred_x0 = self.step(noise_pred, t, latents) 
                latentsuam, pred_x0_uam = self.step(noise_pred_uam, t, latentsuam)

                    
            if noise_loss_list is not None:
                latents = torch.concat((latents[:1]+noise_loss_list[i][:1],latents[1:]))

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list =[self.latent2image(img, return_type="pt") for img in pred_x0_list] 
            pred_x0_uam_list = [self.latent2image(img, return_type="pt") for img in pred_x0_uam_list] 
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list] 
            return image, pred_x0_list, latents_list,pred_x0_uam_list
        return image




    @torch.no_grad()
    def origin_call(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        noise_loss_list=None,edit_stage=True, prox=None, quantile=0.7,
                   image_enc=None, recon_lr=0.1, recon_t=400,
                   inversion_guidance=False, x_stars=None, i=0,
                   dilate_mask=0,
        **kwds):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)

        latents_list = [latents]
        pred_x0_list = [latents]


        prox="l0"
        recon_lr=0

        inversion_guidance=True
        print("w/ mask")
     
        print(f"a quantile:{quantile}, guidance_scale:{guidance_scale}, recon_t:{recon_t}, recon_lr:{recon_lr}")
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
        
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:


                
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2, dim=0)
                step_kwargs = {
                    'ref_image': None,
                    'recon_lr': 0,
                    'recon_mask': None,
                }
                mask_edit = None
                if edit_stage and prox is not None and ((recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t)):
                    if prox == 'l1':
                        x0_delta = noise_prediction_text - noise_pred_uncond
                        if quantile > 0:
                            threshold = x0_delta.abs().quantile(quantile)
                        else:
                            threshold = -quantile  # if quantile is negative, use it as a fixed threshold
                        x0_delta -= x0_delta.clamp(-threshold, threshold)
                        x0_delta = torch.where(x0_delta > 0, x0_delta-threshold, x0_delta)
                        x0_delta = torch.where(x0_delta < 0, x0_delta+threshold, x0_delta)
                        if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                            step_kwargs['ref_image'] = image_enc
                            step_kwargs['recon_lr'] = recon_lr
                            mask_edit = (x0_delta.abs() > threshold).float()
                            if dilate_mask > 0:
                                radius = int(dilate_mask)
                                mask_edit = dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                            step_kwargs['recon_mask'] = 1 - mask_edit
                    elif prox == 'l0':
              
                        if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                            x0_delta = (noise_prediction_text[1]-noise_prediction_text[0]).unsqueeze(0) 
                            if quantile > 0:
                                threshold = x0_delta.abs().quantile(quantile)
                            else:
                                threshold = -quantile  # if quantile is negative, use it as a fixed threshold
                            x0_delta -= x0_delta.clamp(-threshold, threshold) 


                            step_kwargs['ref_image'] = image_enc
                            step_kwargs['recon_lr'] = recon_lr
                            mask_edit = (x0_delta.abs() > threshold).float()
                            if dilate_mask > 0:
                                radius = int(dilate_mask)
                                mask_edit = dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                            step_kwargs['recon_mask'] = 1 - mask_edit

                    else:
                        raise NotImplementedError

                    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)




            latents, pred_x0 = self.step(noise_pred, t, latents)

            

            if mask_edit is not None and inversion_guidance and (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                recon_mask = 1 - mask_edit

                latents = latents - recon_lr * (latents - x_stars[len(x_stars)-i-2].expand_as(latents)) * recon_mask

  
                    
            if noise_loss_list is not None:
                latents = torch.concat((latents[:1]+noise_loss_list[i][:1],latents[1:]))
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, pred_x0_list, latents_list
        return image


    @torch.no_grad()
    def invert_origin(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
        
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))

        latents_list = [latents]
        pred_x0_list = [latents]
        
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, pred_x0_list, latents_list
        return latents,latents_list



    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))

        latents_list1 = [latents]
        pred_x0_list1 = [latents]
        latents_list2 = [latents]
        pred_x0_list2 = [latents]    
        latents1=latents
        latents2=latents

        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):

            model_inputs = torch.cat([latents1,latents2])

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0) 

            # compute the previous noise sample x_t-1 -> x_t
            latents1, pred_x0_1 = self.next_step(noise_pred_uncon, t, latents1)
            latents2, pred_x0_2 = self.next_step(noise_pred_con, t, latents2)
            latents_list1.append(latents1)


        if return_intermediates:
            # return the intermediate laters during inversion
            pred_x0_list1 = None
            pred_x0_list2 = None
            return latents1,latents2, pred_x0_list1,pred_x0_list2, latents_list1
        return latents2,latents_list1

