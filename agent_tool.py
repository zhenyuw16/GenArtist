from PIL import Image
import os
import os.path as osp
import torch
import diffusers
import cv2
from agent_tool_aux import main_aux
from agent_tool_edit import main_edit
from agent_tool_generate import main_generate
import gc
import json


def command_parse(commands, text, text_bg, dir='inputs'):
    args = []
    generation_arg = None
    for i in range(len(commands)):
        command = commands[i]
        if command['tool'] == 'edit':
            k = len(args)
            if 'box' in command:
                if 'intbox' in command:
                    bb = command['box']
                    command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"], 'box': command['box']} }
            else:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"]} }
            args.append(arg)

            arg = {"tool": "object_addition_anydoor",  "text": text, "text_bg": text_bg, 
                    "output": osp.join(dir, str(k+2)+'.png'), "output_mask": osp.join(dir, str(k+2)+'_mask.png'), 
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": command["edit"], "layout": osp.join(dir, str(k+1)+'_mask.png') } }
            args.append(arg)

            arg = {"tool": "replace_anydoor", "output": osp.join(dir, str(k+3)+'.png'), 
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": osp.join(dir, str(k+2)+'.png'),
                             "object_mask": osp.join(dir, str(k+2)+'_mask.png'), "mask": osp.join(dir, str(k+1)+'_mask.png'),  } }
            args.append(arg)
        elif command['tool'] == 'move':
            if 'intbox' in command:
                bb = command['box']
                command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
            k = len(args)
            arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"]} }
            args.append(arg)

            arg = {"tool": "remove", "output":  osp.join(dir, str(k+2)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "mask": osp.join(dir, str(k+1)+'_mask.png')} }
            args.append(arg)

            arg = {"tool": "addition_anydoor", "output": osp.join(dir, str(k+3)+'.png'), 
                "input": {"image": osp.join(dir, str(k+2)+'.png'), "object": osp.join(dir, str(k)+'.png'), 
                        "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "mask": command["box"]  } }
            args.append(arg)
        elif command['tool'] == 'addition':
            k = len(args)
            if 'intbox' in command:
                bb = command['box']
                command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
            arg = {"tool": "object_addition_anydoor",  "text": text, "text_bg": text_bg, 
                    "output": osp.join(dir, str(k+1)+'.png'), "output_mask": osp.join(dir, str(k+1)+'_mask.png'), 
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": command["input"], "layout": command["box"] } }
            args.append(arg)

            arg = {"tool": "addition_anydoor", "output": osp.join(dir, str(k+2)+'.png'), 
                "input": {"image": osp.join(dir, str(k)+'.png'), "object": osp.join(dir, str(k+1)+'.png'), 
                        "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "mask": command["box"]  } }
            args.append(arg)
        elif command['tool'] == 'remove':
            k = len(args)
            arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"], "mask_threshold": command.get("mask_thr", 0.0)} }
            if 'box' in command:
                if 'intbox' in command:
                    bb = command['box']
                    command['box'] = [bb[0]/512., bb[1]/512., (bb[2]-bb[0])/512., (bb[3]-bb[1])/512.]
                arg['input']['box'] = command['box']
            args.append(arg)

            arg = {"tool": "remove", "output":  osp.join(dir, str(k+2)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "mask": osp.join(dir, str(k+1)+'_mask.png')} }
            args.append(arg)
        elif command['tool'] == 'instruction':
            k = len(args)
            arg = {"tool": "instruction", "output": osp.join(dir, str(k+1)+'.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["text"]} }
            args.append(arg)
        elif command['tool'] == 'edit_attribute':
            k = len(args)
            if 'box' in command:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"], 'box': command['box']} }
            else:
                arg = {"tool": "segmentation", "output": osp.join(dir, str(k+1)+'_mask.png'), "input": {"image": osp.join(dir, str(k)+'.png'), "text": command["input"]} }
            args.append(arg)

            arg = {"tool": "attribute_diffedit", 
                    "output": osp.join(dir, str(k+2)+'.png'),  
                    "input": {"image": osp.join(dir, str(k)+'.png'), "object": command["input"], "object_mask": osp.join(dir, str(k+1)+'_mask.png'), "attr": command["text"] } }
            args.append(arg)
        
        elif command['tool'] in ['text_to_image_SDXL', 'image_to_image_SD2', 'layout_to_image_LMD', 'layout_to_image_BoxDiff', 'superresolution_SDXL']:
            generation_arg = command
            generation_arg["output"] = osp.join(dir, '0.png')
    
    k = len(args)
    arg = {"tool": "superresolution_SDXL", "input": {"image": osp.join(dir, str(k)+'.png')}, "output": osp.join(dir, str(k+1)+'.png')}
    args.append(arg)
    if generation_arg is not None:
        args = [generation_arg] + args
    return args



if __name__ == '__main__':

    ### text to image generation
    input_image = None
    text = 'an oil painting, where a green vintage car, a black scooter on the left of it and a blue bicycle on the right of it, are parked near a curb, with three birds in the sky'
    text_bg = 'a realistic scene' #'an oil painting'
    commands = json.load(open('demo/generation_1.json'))


    ### text to image generation
    # input_image = None
    # text = 'Two white sheep on the left, a black goat on the middle and a white goat on the right in a field.'
    # text_bg = 'a realistic scene'
    # commands = json.load(open('demo/generation_2.json'))

    
    ### editing
    # input_image = "inputs/368667-input.png"
    # text = 'A realistic photo'
    # text_bg = 'A realistic photo'
    # # editing: The farm should have a stream, and a giraffe should be on the left of the stream under a supernova explosion sky.
    # commands = json.load(open('demo/editing_1.json')) 


    if input_image is not None:
        os.system('cp ' + input_image + ' inputs/0.png')
        im = cv2.imread('inputs/0.png')
        im = cv2.resize(im, (512,512))
        cv2.imwrite('inputs/0.png', im)

    seq_args = command_parse(commands, text, text_bg,)
    print(seq_args)
    for i in range(len(seq_args)):
        json.dump(seq_args[i], open('input.json','w'))
        if seq_args[i]['tool'] in ["object_addition_anydoor", "segmentation"]:
            # main_aux(seq_args[i])
            os.system('python agent_tool_aux.py --json_out True')
        elif seq_args[i]['tool'] in ["addition_anydoor", "replace_anydoor", "remove", "instruction", "attribute_diffedit"]:
            # main_edit(seq_args[i])
            os.system('python agent_tool_edit.py --json_out True')
        elif seq_args[i]['tool'] in ['text_to_image_SDXL', 'image_to_image_SD2', 'layout_to_image_LMD', 'layout_to_image_BoxDiff', 'superresolution_SDXL']:
            # main_generate(seq_args[i])
            os.system('python agent_tool_generate.py --json_out True')
