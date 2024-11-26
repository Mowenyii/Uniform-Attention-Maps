import torch
import torch.nn.functional as nnf
import abc

from utils.utils import get_word_inds, get_time_words_attention_alpha
from models.p2p import seq_aligner


MAX_NUM_WORDS = 77
LATENT_SIZE = (64, 64)
LOW_RESOURCE = False


score_maps = []
note = False
invert_note = False
name = ""


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




from torch.utils.tensorboard import SummaryWriter





def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None, **kwargs):
            if isinstance(context, dict):  # NOTE: compatible with ELITE (0.11.1)
                context = context["CONTEXT_TENSOR"]
            is_cross = context is not None


            global invert_note, note

            if (
               (invert_note or note )  and x.shape[0]==1
            ):  
                batch_size, sequence_length, dim = x.shape
                h = self.heads
                q = self.to_q(x)

                context = context if is_cross else x

                k = self.to_k(context)
                v = self.to_v(context)

                q = self.reshape_heads_to_batch_dim(
                    q
                ) 
                k = self.reshape_heads_to_batch_dim(k)  
                v = self.reshape_heads_to_batch_dim(v) 
                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

                if mask is not None:
                    mask = mask.reshape(batch_size, -1)
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask = mask[:, None, :].repeat(h, 1, 1)
                    sim.masked_fill_(~mask, max_neg_value)

            
                attn = sim.softmax(dim=-1)
                attn1=attn
                if is_cross:
                    attn1=torch.ones_like(attn,device=attn.device)*(1/77)

                out = torch.einsum("b i j, b j d -> b i d", attn1, v)
                out = self.reshape_batch_dim_to_heads(out)
                out = to_out(out)  

                return out  
            

            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)

            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)

            q = self.reshape_heads_to_batch_dim(
                q
            )  
            k = self.reshape_heads_to_batch_dim(k)  
            v = self.reshape_heads_to_batch_dim(v)


            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            attn = controller(attn, is_cross, place_in_unet)  

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)

            return to_out(out) 

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "CrossAttention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = (
        model.unet.named_children()
    )  
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def get_equalizer(text, word_select, values, tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


class LocalBlend:
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=LATENT_SIZE)
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1 - int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend: 
            if (
                len(attention_store.keys()) == 0
                or attention_store["down_cross"] == []
                or attention_store["up_cross"] == []
            ):
                return x_t
            try:

                maps = (
                    attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
                ) 
                maps = [
                    item.reshape(
                        self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS
                    )
                    for item in maps
                ]
                maps = torch.cat(maps, dim=1)
                mask = self.get_mask(maps, self.alpha_layers, True)
                if self.substruct_layers is not None:
                    maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                    mask = mask * maps_sub
                mask = mask.float()
                x_t = x_t[:1] + mask * (x_t - x_t[:1])
            except:
                return x_t

        return x_t

    def __init__(
        self,
        prompts,
        words,
        substruct_words=None,
        start_blend=0.2,
        th=(0.3, 0.3),
        tokenizer=None,
        device="cuda",
        num_ddim_steps=50,
    ):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0
        self.th = th


class EmptyControl:
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross, place_in_unet,part_attn=None):
        return attn


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
            step_idx if step_idx is not None else []
        )  
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        

    def __call__(self, attn, is_cross, place_in_unet,part_attn=None):

        self.cur_att_layer += 1  

        if (
            self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers
        ): 
            self.cur_att_layer = 0
            self.cur_step += 1
        return attn


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross, place_in_unet):
        raise NotImplementedError

    def __call__(self, attn, is_cross, place_in_unet,part_attn=None):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                if attn.shape == torch.Size([1, 1]):
                    self.forward(attn, is_cross, place_in_unet) 
                    attn = 0 
                else:
                    h = attn.shape[0]
                    attn[h // 2 :] = self.forward(
                        attn[h // 2 :], is_cross, place_in_unet 
                    )
                 
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
           
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(
        self, layer_idx=None, start_step=None, total_steps=None, step_idx=None
    ):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_idx = None


class SpatialReplace(EmptyControl):
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject, num_ddim_steps=50):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * num_ddim_steps)


class AttentionStore(AttentionControl):
    MODEL_TYPE = {"SD": 16, "SDXL": 70}

    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        if attn.shape[1] <= 32**2:  # avoid memory overhead
            if attn.shape == torch.Size([1, 1]): 
                return attn
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    try:
                        if self.step_store[key] != [] and self.step_store[key][
                            i
                        ].shape != torch.Size([1, 1]):
                            self.attention_store[key][i] += self.step_store[key][i]
                    except:
                        pass


        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


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
        super(AttentionStore, self).__init__(
            layer_idx=layer_idx,
            start_step=start_step,
            total_steps=total_steps,
            step_idx=step_idx,
        )
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = (
            layer_idx if layer_idx is not None else []
        )  
        self.step_idx = (
            step_idx if step_idx is not None else []
        ) 


class AttentionControlEdit(AttentionStore, abc.ABC):
    MODEL_TYPE = {"SD": 16, "SDXL": 70}

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32**2:
            attn_base = attn_base.unsqueeze(0).expand(
                att_replace.shape[0], *attn_base.shape
            )
            return attn_base
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross, place_in_unet):  
        super(AttentionControlEdit, self).forward(
            attn, is_cross, place_in_unet
        )  
        if is_cross or (
            self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]
        ):  
            if attn.shape == torch.Size([1, 1]): 
                return attn
            h = attn.shape[0] // (self.batch_size)

            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])

            attn_base, attn_repalce = attn[0], attn[1:]  
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = (
                    self.replace_cross_attention(attn_base, attn_repalce) * alpha_words
                    + (1 - alpha_words) * attn_repalce
                )
                attn[1:] = attn_repalce_new  
            else:
                attn[1:] = self.replace_self_attention(
                    attn_base, attn_repalce, place_in_unet
                )
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    try:
                        if not self.step_store[key][i].shape == torch.Size([1, 1]):
                            self.attention_store[key][i] += self.step_store[key][i]
                    except:
                        pass
        self.step_store = self.get_empty_store()


    def __init__(
        self,
        prompts,
        num_steps,
        cross_replace_steps,
        self_replace_steps,
        local_blend,
        tokenizer=None,
        device="cuda",
        start_step=4,
        start_layer=10,
        layer_idx=None,
        step_idx=None,
        total_steps=50,
        model_type="SD",
    ):
        super(AttentionControlEdit, self).__init__(
            layer_idx=layer_idx,
            start_step=start_step,
            total_steps=total_steps,
            step_idx=step_idx,
        )
        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, tokenizer
        ).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(
            num_steps * self_replace_steps[1]
        )
        self.local_blend = local_blend
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = (
            layer_idx if layer_idx is not None else list(range(6, 7))
        )  
        self.step_idx = (
            step_idx if step_idx is not None else []
        )




class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        prompts,
        num_steps,
        cross_replace_steps,
        self_replace_steps,
        local_blend=None,
        tokenizer=None,
        device="cuda",
        layer_idx=None,
        start_step=None,
        total_steps=None,
        step_idx=None,
    ):
        super(AttentionReplace, self).__init__(
            prompts=prompts,
            num_steps=num_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            local_blend=local_blend,
            device=device,
            layer_idx=layer_idx,
            start_step=start_step,
            total_steps=total_steps,
            step_idx=step_idx,
        )
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
    
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps,
        cross_replace_steps,
        self_replace_steps,
        local_blend=None,
        tokenizer=None,
        device="cuda",
        layer_idx=None,
        start_step=None,
        total_steps=None,
        step_idx=None,
    ):
        super(AttentionRefine, self).__init__(
            prompts=prompts,
            num_steps=num_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            local_blend=local_blend,
            device=device,
            layer_idx=layer_idx,
            start_step=start_step,
            total_steps=total_steps,
            step_idx=step_idx,
        )
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(
                attn_base, att_replace
            )
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]

        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps,
        cross_replace_steps,
        self_replace_steps,
        equalizer,
        local_blend=None,
        controller=None,
        device="cuda",
        layer_idx=None,
        start_step=None,
        total_steps=None,
        step_idx=None,
    ):
        super(AttentionReweight, self).__init__(
            prompts=prompts,
            num_steps=num_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            local_blend=local_blend,
            device=device,
            layer_idx=layer_idx,
            start_step=start_step,
            total_steps=total_steps,
            step_idx=step_idx,
        )
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def make_controller(
    pipeline,
    prompts,
    is_replace_controller,
    cross_replace_steps,
    self_replace_steps,
    blend_words=None,
    equilizer_params=None,
    num_ddim_steps=50,
    device="cuda",
    layer_idx=None,
    start_step=None,
    total_steps=None,
    step_idx=None,
) -> AttentionControlEdit:
    if blend_words is None:
        lb = None

    else:
        lb = LocalBlend(
            prompts,
            blend_words,
            tokenizer=pipeline.tokenizer,
            device=device,
            num_ddim_steps=num_ddim_steps,
        )

    if is_replace_controller:
        controller = AttentionReplace(
            prompts,
            num_ddim_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            local_blend=lb,
            tokenizer=pipeline.tokenizer,
            layer_idx=layer_idx,
            start_step=start_step,
            total_steps=total_steps,
            step_idx=step_idx,
        )
    else:
        controller = AttentionRefine(
            prompts,  
            num_ddim_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            local_blend=lb,
            tokenizer=pipeline.tokenizer,
            layer_idx=layer_idx,
            start_step=start_step,
            total_steps=total_steps,
            step_idx=step_idx,
        )
    if equilizer_params is not None:
        eq = get_equalizer(
            prompts[1],
            equilizer_params["words"],
            equilizer_params["values"],
            tokenizer=pipeline.tokenizer,
        )
        controller = AttentionReweight(
            prompts,  
            num_ddim_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            equalizer=eq,
            local_blend=lb,
            controller=controller,
            layer_idx=layer_idx,
            start_step=start_step,
            total_steps=total_steps,
            step_idx=step_idx,
        )
    return controller
