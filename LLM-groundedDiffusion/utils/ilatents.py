
from models import sam
from models import pipelines
import torch

DEFAULT_SO_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate, two, many, group, occlusion, occluded, side, border, collate"
DEFAULT_OVERALL_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"



def get_all_latents(img_np, models, inv_seed=1):
    generator = torch.cuda.manual_seed(inv_seed)
    cln_latents = pipelines.encode(models.model_dict, img_np, generator)
    # Magic prompt
    # Have tried using the parsed bg prompt from the LLM, but it doesn't work well
    prompt = "A realistic photo of a scene"
    input_embeddings = models.encode_prompts(
        prompts=[prompt],
        tokenizer=models.model_dict.tokenizer,
        text_encoder=models.model_dict.text_encoder,
        negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT,
        one_uncond_input_only=False,
    )
    # Get all hidden latents
    all_latents = pipelines.invert(
        models.model_dict,
        cln_latents,
        input_embeddings,
        num_inference_steps=51,
        guidance_scale=2.5,
    )
    return all_latents, input_embeddings