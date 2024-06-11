from PIL import Image
import os
import torch
import diffusers
import cv2
import numpy as np
import gc
import numpy as np
import random
import argparse
import json
import math

def main_edit(args):
    default_seed = 42
    torch.manual_seed(default_seed)
    np.random.seed(default_seed)
    random.seed(default_seed)

    if args["tool"] == "addition_anydoor":
        os.sys.path.append('./AnyDoor')
        os.sys.path.append('./AnyDoor/dinov2')
        os.sys.path.append('./AnyDoor/dinov2/datasets')
        from run_inference import inference_single_image
        reference_image_path = args["input"]["object"] #
        reference_image_mask_path = args["input"]["object_mask"] 
        bg_image_path = args["input"]["image"]
        bg_mask_path = args["input"]["object_mask"]
        # save_path = './examples/TestDreamBooth/GEN/gen_res.png'

        # reference image + reference mask
        # You could use the demo of SAM to extract RGB-A image with masks
        # https://segment-anything.com/demo
        image = cv2.imread( reference_image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(reference_image_mask_path)
        ref_image = image
        ref_mask = mask[:,:,0] / 255.

        # background image
        back_image = cv2.imread(bg_image_path).astype(np.uint8)
        back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

        # background mask 
        tar_mask = np.zeros((512, 512)) #cv2.imread(bg_mask_path)[:,:,0]
        bb = [int(i*512) for i in args["input"]["mask"]]
        tar_mask[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] =1 # 255
        tar_mask = tar_mask.astype(np.uint8)

        # print(ref_image.shape, ref_mask.shape, back_image.shape, tar_mask.shape)

        gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
        h,w = back_image.shape[0], back_image.shape[0]
        ref_image = cv2.resize(ref_image, (w,h))
        vis_image = cv2.hconcat([ref_image, back_image, gen_image])

        cv2.imwrite(args["output"], gen_image[:,:,::-1])
    
    elif args["tool"] == 'attribute_diffedit':
        from diffusers import StableDiffusionDiffEditPipeline
        from diffusers import DDIMScheduler, DDIMInverseScheduler
        from PIL import Image

        img_url = args["input"]["image"]
        mask_prompt = args["input"]["object"]
        prompt = args["input"]["attr"]

        init_image = Image.open(img_url).resize((768, 768))

        pipe = StableDiffusionDiffEditPipeline.from_pretrained(
            "stable-diffusion-2", torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        mask_image = np.array(Image.open(args["input"]["object_mask"]).resize((96,96))) / 255.
        mask_image = mask_image[None,:,:]
        image_latents = pipe.invert(image=init_image, prompt=mask_prompt, inpaint_strength=0.8).latents
        image = pipe(prompt=prompt, mask_image=mask_image, image_latents=image_latents, 
                    guidance_scale=10.5, inpaint_strength=0.8).images[0].resize((512, 512))

        image.save(args["output"])
    
    elif args["tool"] == "replace_anydoor":
        os.sys.path.append('./AnyDoor')
        os.sys.path.append('./AnyDoor/dinov2')
        os.sys.path.append('./AnyDoor/dinov2/datasets')
        from run_inference import inference_single_image
        reference_image_path = args["input"]["object"] #
        reference_image_mask_path = args["input"]["object_mask"] 
        bg_image_path = args["input"]["image"]
        bg_mask_path = args["input"]["mask"]
        # save_path = './examples/TestDreamBooth/GEN/gen_res.png'

        # reference image + reference mask
        # You could use the demo of SAM to extract RGB-A image with masks
        # https://segment-anything.com/demo
        image = cv2.imread( reference_image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(reference_image_mask_path)
        ref_image = image
        ref_mask = mask[:,:,0] / 255.

        # y, x = np.nonzero(ref_mask)
        # ref_bbox = [x.min(), y.min(), x.max()-x.min(), y.max()-y.min()]
        # ref_bbox = [k/512. for k in ref_bbox]
        # ref_mask = np.zeros((512, 512)) #cv2.imread(bg_mask_path)[:,:,0]
        # bb = [int(i*512) for i in ref_bbox]
        # ref_mask[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] = 1 #255
        # ref_mask = ref_mask.astype(np.uint8)

        if type(args["input"]["mask"]) is str:
            mask_image = cv2.imread(args["input"]["mask"])[:,:,0]
            y, x = np.nonzero(mask_image)
            args["input"]["mask"] = [x.min(), y.min(), x.max()-x.min(), y.max()-y.min()]
            args["input"]["mask"] = [k/512. for k in args["input"]["mask"]]

        # background image
        back_image = cv2.imread(bg_image_path).astype(np.uint8)
        back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

        # background mask 
        # tar_mask = cv2.imread(bg_mask_path)[:,:,0]
        # tar_mask = tar_mask.astype(np.uint8)
        tar_mask = np.zeros((512, 512)) #cv2.imread(bg_mask_path)[:,:,0]
        bb = [int(i*512) for i in args["input"]["mask"]]
        tar_mask[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] = 1. #255
        tar_mask = tar_mask.astype(np.uint8)

        # print(ref_image.shape, ref_mask.shape, back_image.shape, tar_mask.shape)

        gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
        h,w = back_image.shape[0], back_image.shape[0]
        ref_image = cv2.resize(ref_image, (w,h))
        vis_image = cv2.hconcat([ref_image, back_image, gen_image])

        cv2.imwrite(args["output"], gen_image[:,:,::-1])
    
    elif args["tool"] == "remove":
        os.sys.path.append('./Inpaint-Anything')
        from lama_inpaint import inpaint_img_with_lama
        from utils import load_img_to_array, save_array_to_img, dilate_mask, show_mask, show_points, get_clicked_point
        from PIL import Image
        img = load_img_to_array(args["input"]["image"])
        masks = np.array(Image.open(args["input"]["mask"])) #[None,:,:]
        masks = masks.astype(np.uint8) #* 255
        # masks = dilate_mask(masks, 15)
        masks = dilate_mask(masks, 10)
        # print(img.shape, masks.shape)
        img_inpainted = inpaint_img_with_lama(
            img, masks, './Inpaint-Anything/lama/configs/prediction/default.yaml', './Inpaint-Anything/pretrained_models/big-lama', device='cuda')
        # Image.fromarray(img_inpainted).save(args["output"])
        save_array_to_img(img_inpainted, args["output"])
    
    elif args["tool"] == "instruction":
        from omegaconf import OmegaConf
        from PIL import Image, ImageOps
        from torch import autocast
        import sys
        import k_diffusion as K
        from einops import rearrange
        os.sys.path.append("./instruct-pix2pix/")
        from edit_cli import CFGDenoiser, load_model_from_config

        sys.path.append("./instruct-pix2pix/stable_diffusion")
        step = 50
        config = OmegaConf.load("./instruct-pix2pix/configs/generate.yaml")
        # model = load_model_from_config(config, "./instruct-pix2pix/checkpoints/MagicBrush-epoch-52-step-4999.ckpt", None)
        # model = load_model_from_config(config, "./instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt", None)
        model = load_model_from_config(config, "./instruct-pix2pix/checkpoints/MagicBrush-epoch-000168.ckpt", None)
        model.eval().cuda()
        model_wrap = K.external.CompVisDenoiser(model)
        model_wrap_cfg = CFGDenoiser(model_wrap)
        null_token = model.get_learned_conditioning([""])

        seed = 666 #42 #random.randint(0, 100000) if args.seed is None else args.seed
        input_image = Image.open(args["input"]["image"]).convert("RGB")
        width, height = input_image.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning([args["input"]["text"]])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
            cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(step)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": 7.5,
                "image_cfg_scale": 1.5,
            }
            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        edited_image.save(args["output"])



    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="for agent")
    parser.add_argument("--json_out", type=bool, default=False, help="Path to the json file")
    args = parser.parse_args()

    if args.json_out:
        args = json.load(open('input.json'))
    else:
        args = {'tool': 'instruction', 'output': 'inputs/368667-input-output.png', 'input': {'image': 'inputs/368667-input.png', 'text': 'The farm should have a stream, and a giraffe should be on the left of the stream under a supernova explosion sky.'}}
    main_edit(args)


    

