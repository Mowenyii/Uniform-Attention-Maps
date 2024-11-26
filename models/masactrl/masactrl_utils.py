
import torch
import torch.nn as nn

from einops import rearrange, repeat
import pdb

score_maps = []
note = False
invert_note = False
name = ""
MAX_NUM_WORDS = 77
LATENT_SIZE = (64, 64)
LOW_RESOURCE = False


def get_invert_note():
    global invert_note
    return invert_note


def set_invert_note():
    global invert_note
    invert_note = True


def clear_invert_note():
    global invert_note
    invert_note = False


def set_name(n=""):
    global name
    name = n


def get_global_score_maps():
    global score_maps
    return score_maps


def get_note():
    global note
    return note


def set_note():
    global note
    note = True


def clear_note():
    global note
    note = False


def clear_global_score_maps():
    clear_note()
    global score_maps, note
    score_maps = []

    



class MyEmptyControl:
    MODEL_TYPE = {"SD": 16, "SDXL": 70}

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    def __init__(
        self,
        device="cuda",
        start_step=4,
        start_layer=10,
        layer_idx=None,
        step_idx=None,
        total_steps=50,
        model_type="SD",
    ):

        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = (
            layer_idx if layer_idx is not None else []
        )  
        self.step_idx = (
            step_idx if step_idx is not None else list(range(start_step, total_steps))
        ) 
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
   

    def __call__(self, attn, is_cross, place_in_unet,part_attn=None, **kwargs):

        self.cur_att_layer += 1  
        if (
            self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers
        ):  
            self.cur_att_layer = 0
            self.cur_step += 1
        
        return attn



class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
          
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v) 
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]
        self.self_attns_step.clear()
        self.cross_attns_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)




def regiter_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask
            is_cross = context is not None

            global invert_note, note

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)

            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)

            
            if (
                (invert_note) and hasattr(editor, "step_idx") and not is_cross 
            ): 

                out = torch.einsum('b i j, b j d -> b i d', attn, v) 
                out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
                out = to_out(out)
       
                try:
                    editor(
                            torch.zeros((1, 1), device=x.device), is_cross, place_in_unet 
                        )  
                except:
                    print(editor) 
                    pdb.set_trace()
                return out  
            elif (
                invert_note and hasattr(editor, "step_idx") and is_cross 
            ):  
                if attn.shape[0]==8 : 
                    attn1=torch.ones_like(attn,device=attn.device)*(1/77)
   
                    out1 = torch.einsum('b i j, b j d -> b i d', attn1, v) 
                    out1 = rearrange(out1, '(b h) n d -> b n (h d)', h=self.heads)
                    out1 = to_out(out1)
      
                    editor(
                        torch.zeros((1, 1), device=x.device), is_cross, place_in_unet 
                    )  
                    return out1                
                elif attn.shape[0]==16: 
                    out = torch.einsum('b i j, b j d -> b i d', attn, v) 
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
                    out = to_out(out)
                    
                    attn1=torch.ones_like(attn,device=attn.device)*(1/77)
                    out1 = torch.einsum('b i j, b j d -> b i d', attn1, v) 
                    out1 = rearrange(out1, '(b h) n d -> b n (h d)', h=self.heads)
                    out1 = to_out(out1)
            
                    editor(
                        torch.zeros((1, 1), device=x.device), is_cross, place_in_unet 
                    )  
                    try:
                        fin_out = torch.cat(
                            [
                                out1[0].unsqueeze(0), 
                                out[1].unsqueeze(0), 
                            ]
                        )  
                    except:
                        pdb.set_trace()    
                    return fin_out
                 
                elif attn.shape[0]==32:
                    out = torch.einsum('b i j, b j d -> b i d', attn, v) 
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
                    out = to_out(out)     
                    attn1=torch.ones_like(attn,device=attn.device)*(1/77) 
                    out1 = torch.einsum('b i j, b j d -> b i d', attn1, v) 
                    out1 = rearrange(out1, '(b h) n d -> b n (h d)', h=self.heads)
                    out1 = to_out(out1)
                    print(f"invert cros3: out1:{out1.shape} step:{editor.cur_step},layer:{editor.cur_att_layer} {editor.cur_att_layer//2} attn1:{attn.shape}")
                    editor(
                        torch.zeros((1, 1), device=x.device), is_cross, place_in_unet 
                    )                
                    try:
                        fin_out = torch.cat(
                            [
                                out1[0].unsqueeze(0), 
                                out[1].unsqueeze(0), 
                                out1[2].unsqueeze(0), 
                                out[-1].unsqueeze(0), 
                            ]
                        )  
                    except:
                        pdb.set_trace()    
                    return fin_out
                



            if (
                (note ) and hasattr(editor, "step_idx") and is_cross 
            ): 
                if attn.shape[0]==8:
                    attn1=torch.ones_like(attn,device=attn.device)*(1/77)
             
                    out1 = torch.einsum('b i j, b j d -> b i d', attn1, v) 
                    out1 = rearrange(out1, '(b h) n d -> b n (h d)', h=self.heads)
                    out1 = to_out(out1)
  
                    return out1
                if attn.shape[0]==32:

                    out = editor(q, k, v, sim, attn, is_cross, place_in_unet,self.heads, scale=self.scale)
                    
    
                    out = to_out(out)
                 
                    return out
                    

                elif attn.shape[0]==40:

                    attn1=torch.ones_like(attn,device=attn.device)*(1/77) 
                    out1 = torch.einsum('b i j, b j d -> b i d', attn1, v) 
                    out1 = rearrange(out1, '(b h) n d -> b n (h d)', h=self.heads)
                    out1 = to_out(out1)
                    out = editor(q, k, v, sim, attn, is_cross, place_in_unet,self.heads, scale=self.scale)
                    
             
                    out = to_out(out)
                   
                    try:
                        fin_out = torch.cat(
                            [
                                out1[0].unsqueeze(0), 
                                out[1].unsqueeze(0), 
                                out1[2].unsqueeze(0), 
                                out[-1].unsqueeze(0), 
                            ]
                        )  
                    except:
                        pdb.set_trace()              
                    return fin_out  

            else:
                if not is_cross and attn.shape[0]==8:
                   
                    out = torch.einsum('b i j, b j d -> b i d', attn, v) 
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
                    out = to_out(out)
                    
                    return out  
                else:
                   
                    try:
                        out = editor(q, k, v, sim, attn, is_cross, place_in_unet,self.heads, scale=self.scale)
                    except:
                        pdb.set_trace()
               
                return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count



def regiter_attention_editor_ldm(model, editor: AttentionBase):
    """
    Register a attention editor to Stable Diffusion model, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'CrossAttention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.model.diffusion_model.named_children():
        if "input" in net_name:
            cross_att_count += register_editor(net, 0, "input")
        elif "middle" in net_name:
            cross_att_count += register_editor(net, 0, "middle")
        elif "output" in net_name:
            cross_att_count += register_editor(net, 0, "output")
    editor.num_att_layers = cross_att_count
