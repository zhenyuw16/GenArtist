from PIL import Image
import os
import torch
import diffusers
import cv2
import numpy as np
import random
import json
import gc
import argparse

class_lvis = json.load(open('cname7k.json'))

def clear_globals():
    global_vars = globals()
    for var_name in list(global_vars.keys()):
        if var_name != "clear_globals": 
            print(var_name)
            del global_vars[var_name]

def main_aux(args):
    if args["tool"] == "object_addition_anydoor":
        import cv2
        if args["input"]["layout"] == "TBG":
            args["input"]["layout"] = args_layout
        if type(args["input"]["layout"]) is str:
            mask_image = cv2.imread(args["input"]["layout"])[:,:,0]
            y, x = np.nonzero(mask_image)
            args["input"]["layout"] = [x.min(), y.min(), x.max()-x.min(), y.max()-y.min()]
            args["input"]["layout"] = [k/512. for k in args["input"]["layout"]]
        else:
            mask_image = np.zeros((512,512))
            if args.get("attr", False):
                bb = [int(i*512) for i in args["input"]["layout"]]
                mask_image[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] = 255.
        
        
        os.sys.path.append('./LLM-groundedDiffusion/')
        import models
        from models import sam
        from models.sam import sam_box_input
        from utils import parse, utils
        import cv2

        # Load models
        models.sd_key = "diffusers-generation-text-box"
        models.sd_version = "sdv1.4"
        # models.sd_key = "stable-diffusion-2-1-base"
        # models.sd_version = "sdv2"

        models.model_dict = models.load_sd(
            key=models.sd_key,
            # use_fp16=True,
            load_inverse_scheduler=True,
            scheduler_cls=None
        )
        sam_model_dict = sam.load_sam()
        models.model_dict.update(sam_model_dict)

        import image_generator
        from utils.ilatents import get_all_latents

        # Reset random seeds
        default_seed = 50 #666 #42
        torch.manual_seed(default_seed)
        np.random.seed(default_seed)
        random.seed(default_seed)

        image_source = np.array(Image.open(args["input"]["image"]))
        # Background latent preprocessing
        all_latents, _ = get_all_latents(image_source, models, default_seed)
        addition_objs = (args["input"]["object"], args["input"]["layout"])
        spec = {'add_objects': [addition_objs]}
        spec['remove_region'] = cv2.resize(mask_image, (64,64))
        spec['move_objects'] = []
        spec['change_objects'] = []
        spec['prompt'] = args['text'] #'An oil painting at the beach of a blue bicycle to the left of a bench and to the right of a palm tree with three seagulls in the sky'
        spec['bg_prompt'] = args['text_bg'] # 'An oil painting at the beach of a blue bicycle to the left of a bench and to the right of a palm tree with three seagulls in the sky'
        spec["extra_neg_prompt"] = None
        output = image_generator.run_singleobj(
            spec,
            fg_seed_start=default_seed,
            bg_seed=default_seed,
            bg_all_latents=all_latents,
            frozen_step_ratio=0.5,
        )

        layout = [i * 512 for i in args["input"]["layout"]]
        layout[2] = layout[2] + layout[0]
        layout[3] = layout[3] + layout[1]
            
        output.save(args["output"])
        from transformers import SamModel, SamProcessor
        # model = SamModel.from_pretrained("sam-vit-huge").to("cuda")
        # processor = SamProcessor.from_pretrained("sam-vit-huge")
        model = SamModel.from_pretrained("sam-vit-base").to("cuda")
        processor = SamProcessor.from_pretrained("sam-vit-base")
        raw_image = output
        input_boxes = [[layout]]
        inputs = processor([raw_image], input_boxes=input_boxes, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        masks = np.transpose(masks[0][0], (1,2,0))
        masks = masks[:,:,0]
        cv2.imwrite(args["output_mask"], masks.numpy() * 255)
        
    
    elif args["tool"] == "detection":
        if args["input"]["text"] == "TBG":
            args["input"]["text"] = " . ".join(class_lvis)
        from groundingdino.util.inference import load_model, load_image, predict, annotate
        import cv2
        from groundingdino.util import box_ops

        model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")
        IMAGE_PATH = args['input']['image']
        TEXT_PROMPT = args['input']['text']
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

        image_source, image = load_image(IMAGE_PATH)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes[:,2] = boxes[:,2] - boxes[:,0]
        boxes[:,3] = boxes[:,3] - boxes[:,1]
        boxes = np.around(boxes.numpy(), decimals=2).tolist()
        # str_objs = ''
        # for i in range(len(phrases)):
        #     str_objs += phrases[i] + ': '
        #     # str_objs += str(boxes[i]) + ', '
        #     str_objs += str([round(num, 2) for num in boxes[i]]) + ', '
        str_objs = []
        for i in range(len(phrases)):
            str_objs.append((phrases[i], [round(num, 2) for num in boxes[i]]))
        # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # cv2.imwrite("annotated_image.jpg", annotated_frame)
        print(str_objs)
    
    elif args["tool"] == "segmentation":
        from groundingdino.util.inference import load_model, load_image, predict, annotate
        import cv2
        from groundingdino.util import box_ops

        if not "box" in args["input"]:
            model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")
            IMAGE_PATH = args['input']['image']
            TEXT_PROMPT = args['input']['text']
            BOX_TRESHOLD = 0.35
            TEXT_TRESHOLD = 0.25

            image_source, image = load_image(IMAGE_PATH)
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            masks = torch.tensor([])
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        else:
            IMAGE_PATH = args['input']['image']
            boxes = torch.tensor([args["input"]["box"]])
            boxes[:,2] = boxes[:,2] + boxes[:,0]
            boxes[:,3] = boxes[:,3] + boxes[:,1]

        print(boxes)
        if len(boxes) > 0:
            from transformers import SamModel, SamProcessor
            # model = SamModel.from_pretrained("sam-vit-huge", torch_dtype=torch.float16).to("cuda")
            # processor = SamProcessor.from_pretrained("sam-vit-huge", torch_dtype=torch.float16)
            model = SamModel.from_pretrained("sam-vit-base").to("cuda")
            processor = SamProcessor.from_pretrained("sam-vit-base")
            raw_image = Image.open(IMAGE_PATH)
            input_boxes = [boxes.numpy().tolist()]
            for i in range(len(input_boxes[0])):
                bb = [k * 512 for k in input_boxes[0][i]]  ##############
                input_boxes[0][i] = bb
            inputs = processor([raw_image], input_boxes=input_boxes, return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            # masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            # print('aaaaaaaaaaaaaaaaa',args, args.get('mask_threshold', 0.0))
            masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu(), mask_threshold=args['input'].get('mask_threshold', 0.0))
            print('aaaaaaaaaaa', len(masks))
            masks = masks[0]
            
            masks = np.transpose(masks[0], (1,2,0))   ################### index 1
            # masks = np.transpose((masks[0]+masks[1]) > 0.5, (1,2,0)) 
            masks = masks[:,:,0]
            _ = cv2.imwrite(args["output"], masks.numpy() * 255)
    
    # clear_globals()
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
        args = {"tool": "detection", "input": {"image": "./inputs/0.png", "text": "car . person"} }
    main_aux(args)
    
