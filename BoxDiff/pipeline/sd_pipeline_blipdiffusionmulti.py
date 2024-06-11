
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import math

from transformers import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from transformers.modeling_outputs import BaseModelOutputWithPooling

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor

#from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring
# from diffusers.pipelines.utils import randn_tensor
from diffusers.pipelines import BlipDiffusionPipeline, ImagePipelineOutput

from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention
from utils.utils import prepare_cond_image
from diffusers.pipelines.blip_diffusion.blip_image_processing import BlipImageProcessor
from diffusers.pipelines.blip_diffusion.modeling_blip2 import Blip2QFormerModel
# from diffusers.pipelines.blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel
from modeling_ctx_clip import ContextCLIPTextModel

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers.pipelines import BlipDiffusionPipeline
        >>> from diffusers.utils import load_image
        >>> import torch

        >>> blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
        ...     "Salesforce/blipdiffusion", torch_dtype=torch.float16
        ... ).to("cuda")


        >>> cond_subject = "dog"
        >>> tgt_subject = "dog"
        >>> text_prompt_input = "swimming underwater"

        >>> cond_image = load_image(
        ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/dog.jpg"
        ... )
        >>> guidance_scale = 7.5
        >>> num_inference_steps = 25
        >>> negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


        >>> output = blip_diffusion_pipe(
        ...     text_prompt_input,
        ...     cond_image,
        ...     cond_subject,
        ...     tgt_subject,
        ...     guidance_scale=guidance_scale,
        ...     num_inference_steps=num_inference_steps,
        ...     neg_prompt=negative_prompt,
        ...     height=512,
        ...     width=512,
        ... ).images
        >>> output[0].save("image.png")
        ```
"""

class CtxCLIPTextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CtxCLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ctx_embeddings: torch.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=ctx_begin_pos,
        )

        bsz, seq_len = input_shape
        if ctx_embeddings is not None:
            seq_len += ctx_embeddings.size(1)
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=input_ids.device),
            input_ids.to(torch.int).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


class CtxCLIPTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        ctx_embeddings: torch.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if ctx_embeddings is None:
            ctx_len = 0
        else:
            ctx_len = ctx_embeddings.shape[1]

        seq_length = (input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]) + ctx_len

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

            # for each input embeddings, add the ctx embeddings at the correct position
            input_embeds_ctx = []
            bsz = inputs_embeds.shape[0]

            if ctx_embeddings is not None:
                for i in range(bsz):
                    cbp = ctx_begin_pos[i]

                    prefix = inputs_embeds[i, :cbp]
                    # remove the special token embedding
                    suffix = inputs_embeds[i, cbp:]

                    input_embeds_ctx.append(torch.cat([prefix, ctx_embeddings[i], suffix], dim=0))

                inputs_embeds = torch.stack(input_embeds_ctx, dim=0)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class MultiBlipDiffusionPipeline(BlipDiffusionPipeline):
    """
    Pipeline for Zero-Shot Subject Driven Generation using Blip Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer ([`CLIPTokenizer`]):
            Tokenizer for the text encoder
        text_encoder ([`ContextCLIPTextModel`]):
            Text encoder to encode the text prompt
        vae ([`AutoencoderKL`]):
            VAE model to map the latents to the image
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        scheduler ([`PNDMScheduler`]):
             A scheduler to be used in combination with `unet` to generate image latents.
        qformer ([`Blip2QFormerModel`]):
            QFormer model to get multi-modal embeddings from the text and image.
        image_processor ([`BlipImageProcessor`]):
            Image Processor to preprocess and postprocess the image.
        ctx_begin_pos (int, `optional`, defaults to 2):
            Position of the context token in the text encoder.
    """

    model_cpu_offload_seq = "qformer->text_encoder->unet->vae"

    # def __init__(
    #     self,
    #     tokenizer: CLIPTokenizer,
    #     text_encoder: ContextCLIPTextModel,
    #     vae: AutoencoderKL,
    #     unet: UNet2DConditionModel,
    #     scheduler: PNDMScheduler,
    #     qformer: Blip2QFormerModel,
    #     image_processor: BlipImageProcessor,
    #     ctx_begin_pos: int = 2,
    #     mean: List[float] = None,
    #     std: List[float] = None
    # ):
    #     super().__init__(tokenizer, text_encoder, vae, unet, scheduler, qformer, image_processor, ctx_begin_pos, mean, std)

    #     # self.register_modules(
    #     #     tokenizer=tokenizer,
    #     #     text_encoder=text_encoder,
    #     #     vae=vae,
    #     #     unet=unet,
    #     #     scheduler=scheduler,
    #     #     qformer=qformer,
    #     #     image_processor=image_processor,
    #     # )
    #     # self.register_to_config(ctx_begin_pos=ctx_begin_pos, mean=mean, std=std)

    def get_query_embeddings(self, input_image, src_subject):
        return self.qformer(image_input=input_image, text_input=src_subject, return_dict=False)
    
    def text_encoder_multiforward(self, input_ids, ctx_embeddings, ctx_begin_pos):

        # return self.text_encoder.text_model(
        #     ctx_embeddings=ctx_embeddings,
        #     ctx_begin_pos=ctx_begin_pos,
        #     input_ids=input_ids
        # )


        output_attentions = self.text_encoder.text_model.config.output_attentions
        output_hidden_states = self.text_encoder.text_model.config.output_hidden_states
        return_dict = self.text_encoder.text_model.config.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # hidden_states = self.text_encoder.text_model.embeddings(
        #     input_ids=input_ids,
        #     position_ids=None,
        #     ctx_embeddings=ctx_embeddings,
        #     ctx_begin_pos=ctx_begin_pos,
        # )
        ctx_len = ctx_embeddings.shape[1] * len(ctx_begin_pos)##########
        seq_length = input_ids.shape[-1] + ctx_len
        position_ids = self.text_encoder.text_model.embeddings.position_ids[:, :seq_length]

        inputs_embeds = self.text_encoder.text_model.embeddings.token_embedding(input_ids)

        # for each input embeddings, add the ctx embeddings at the correct position
        input_embeds_ctx = []
        bsz = len(ctx_begin_pos) 
        for i in range(bsz):
            cbp = ctx_begin_pos[i]
            # prefix = inputs_embeds[i, :cbp]
            # # remove the special token embedding
            # suffix = inputs_embeds[i, cbp:]

            prefix = inputs_embeds[0, :cbp]
            suffix = inputs_embeds[0, cbp:]
            
            # print(inputs_embeds.shape, cbp, prefix.shape, suffix.shape)

            print(inputs_embeds.shape, prefix.shape, ctx_embeddings.shape, suffix.shape)
            if len(ctx_begin_pos) > 1:
                inputs_embeds = torch.cat([prefix, ctx_embeddings[i], suffix], dim=0).unsqueeze(0)
            else:
                input_embeds_ctx.append(
                    torch.cat([prefix, ctx_embeddings[i], suffix], dim=0)
                )

        if len(ctx_begin_pos) == 1:
            inputs_embeds = torch.stack(input_embeds_ctx, dim=0)

        position_embeddings = self.text_encoder.text_model.embeddings.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        hidden_states = embeddings

        bsz, seq_len = input_shape
        if ctx_embeddings is not None:
            if len(ctx_begin_pos) > 1:
                seq_len += ctx_embeddings.size(1) * len(ctx_begin_pos)
            else:
                seq_len += ctx_embeddings.size(1)
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self.text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )

        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_encoder.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=input_ids.device),
            input_ids.to(torch.int).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    # Copied from diffusers.pipelines.consistency_models.pipeline_consistency_models.ConsistencyModelPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def encode_prompt(self, query_embeds, prompt, device=None):
        device = device or self._execution_device

        # embeddings for prompt, with query_embeds as context
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        max_len -= self.qformer.config.num_query_tokens * query_embeds.shape[0]

        tokenized_prompt = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        # batch_size = query_embeds.shape[0]
        # ctx_begin_pos = [self.config.ctx_begin_pos] * batch_size
        ctx_begin_pos = self._CTX_BEGIN_POS

        # self.text_encoder.text_model.embeddings = CtxCLIPTextEmbeddings(self.text_encoder.text_model.config).to(device)
        # self.text_encoder.text_model = CtxCLIPTextTransformer(self.text_encoder.config).to(device)

        # text_embeddings = self.text_encoder(
        #     input_ids=tokenized_prompt.input_ids,
        #     ctx_embeddings=query_embeds,
        #     ctx_begin_pos=ctx_begin_pos,
        # )[0]
        text_embeddings = self.text_encoder_multiforward(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=ctx_begin_pos,
        )[0]

        return text_embeddings

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: List[str],
        reference_image,
        cldm_cond_image,
        source_subject_category: List[str],
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        neg_prompt: Optional[str] = "",
        prompt_strength: float = 1.0,
        prompt_reps: int = 20,
        bbox: List = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            reference_image (`PIL.Image.Image`):
                The reference image to condition the generation on.
            source_subject_category (`List[str]`):
                The source subject category.
            target_subject_category (`List[str]`):
                The target subject category.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by random sampling.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            height (`int`, *optional*, defaults to 512):
                The height of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            neg_prompt (`str`, *optional*, defaults to ""):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_strength (`float`, *optional*, defaults to 1.0):
                The strength of the prompt. Specifies the number of times the prompt is repeated along with prompt_reps
                to amplify the prompt.
            prompt_reps (`int`, *optional*, defaults to 20):
                The number of times the prompt is repeated along with prompt_strength to amplify the prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
        device = self._execution_device

        # reference_image = self.image_processor.preprocess(
        #     reference_image, image_mean=self.config.mean, image_std=self.config.std, return_tensors="pt"
        # )["pixel_values"]
        # reference_image = reference_image.to(device)
        if type(reference_image[0]) is list:
            reference_images = []
            for i in range(len(reference_image)):
                cc = [self.image_processor.preprocess(k, image_mean=self.config.mean, image_std=self.config.std, return_tensors="pt")["pixel_values"] for k in reference_image[i]]
                cc = [i.to(device) for i in cc]
                reference_images.append(cc)
            reference_image = reference_images
        else:
            reference_image = [self.image_processor.preprocess(i, image_mean=self.config.mean, image_std=self.config.std, return_tensors="pt")["pixel_values"] for i in reference_image]
            reference_image = [i.to(device) for i in reference_image]

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(source_subject_category, str):
            source_subject_category = [source_subject_category]

        batch_size = len(prompt)
        for i in range(batch_size):
            prompt[i] = prompt[i] * prompt_reps

        if type(reference_image[0]) is list:
            query_embeds = []
            for i in range(len(reference_image)):
                query_embeds_0 = torch.cat([self.get_query_embeddings(k, source_subject_category[i]) for k in reference_image[i]])
                query_embeds_0 = torch.mean(query_embeds_0, 0, keepdim=True)
                query_embeds.append(query_embeds_0)
            query_embeds = torch.cat(query_embeds)
        else:
            query_embeds = torch.cat([self.get_query_embeddings(reference_image[i], source_subject_category[i]) for i in range(len(reference_image))])
            
        # print(query_embeds.shape, self._SUB_BEGIN_POS, self._SUB_END_POS)
        # query_embeds = self.get_query_embeddings(reference_image, source_subject_category)
        text_embeddings = self.encode_prompt(query_embeds, prompt, device)
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(device),
                ctx_embeddings=None,
            )[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        scale_down_factor = 2 ** (len(self.unet.config.block_out_channels) - 1)
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self.unet.config.in_channels,
            height=height // scale_down_factor,
            width=width // scale_down_factor,
            generator=generator,
            latents=latents,
            dtype=self.unet.dtype,
            device=device,
        )
        # set timesteps
        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)


        if type(cldm_cond_image) is list and len(cldm_cond_image) > 0:
            cldm_cond_image_both = 1 - math.prod([1 - i/255. for i in cldm_cond_image])
            cldm_cond_image_list = [i/255. for i in cldm_cond_image]
            cldm_cond_image = Image.fromarray(cldm_cond_image_both * 255).convert("RGB")
            cldm_cond_image_list = [torch.tensor(i, device=text_embeddings.device) for i in cldm_cond_image_list]

            # cond_image_both = cond_image[-1]  ############ mask
            # cond_image_list = [i/255. for i in cond_image[0:-1]]
            # cond_image = Image.fromarray(cond_image_both).convert("RGB")
            # cond_image_list = [torch.tensor(i, device=text_embeddings.device) for i in cond_image_list]
        else:
            cldm_cond_image = []
            cldm_cond_image_list = []



        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            do_classifier_free_guidance = guidance_scale > 1.0

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            if hasattr(self, "controlnet") and type(cldm_cond_image) is not list:       ####################################
                cldm_cond_image = prepare_cond_image(
                    cldm_cond_image, width, height, batch_size=1, device=self.device, do_classifier_free_guidance=False
                )
                # cond_image[0] = 0
                # cond_image = [cond_image[0]]
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=cldm_cond_image,
                    # conditioning_scale=0.5, #controlnet_condition_scale,
                    return_dict=False,
                )
            else:
                down_block_res_samples, mid_block_res_sample = None, None

            
            mask = text_embeddings.new_zeros((512, 512, 77))
            postive_value = 2.5
            negative_value = -1e10
        
            # for i in range(77):
            #     mask[:, :, i] = cond_image_both * postive_value + (1 - cond_image_both) * negative_value
            # for k in range(len(cond_image_list)):
            
            for k in range(len(self._SUB_BEGIN_POS)):
                for i in range(self._SUB_BEGIN_POS[k] + int(16 * k /self.visual_downsampling), min(self._SUB_END_POS[k] + int(16 * (k+1)/self.visual_downsampling) - 1, 77)): #range(2, 19+1):
                # for i in range(self._CTX_BEGIN_POS[k], self._CTX_BEGIN_POS[k]+1):       ####################################
                    mask[:, :, i] = cldm_cond_image_list[k] * postive_value + (1 - cldm_cond_image_list[k]) * negative_value

            # for k in range(1):
            #     for i in range(2,18):       ####################################
            #     # for i in range(10, 26): 
            #         mask[:, :, i] = cond_image_list[k] * postive_value + (1 - cond_image_list[k]) * negative_value
            hidden_states = {}
            hidden_states['text_embeddings'] = text_embeddings
            hidden_states['mask'] = mask
            # hidden_states['sigma'] = sigma

            

            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=hidden_states, #text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)