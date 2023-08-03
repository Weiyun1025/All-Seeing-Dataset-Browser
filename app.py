import os
import re
import uuid
import random
import json
import shutil
import requests
import argparse
from pathlib import Path
import dataclasses
from io import BytesIO
from functools import partial
from typing import Any, List , Dict, Union, Literal,TypedDict

import cv2
import numpy as np
import gradio as gr
from PIL import Image
import gradio.themes.base as ThemeBase
from gradio.themes.utils import colors, fonts, sizes
from utils import draw_points_to_image, in_rectangle

# IMAGE_PATH = "/mnt/petrelfs/share_data/huangzhenhang/tmp/as_demo_data/sa_img_000000/"
# IMAGE_PATH = "/mnt/petrelfs/share_data/gaozhangwei/as_demo_data/saved_images"
IMAGE_PATH = "./images"
METAFILE_PATH = "./metafile/metafile.json"
SAVE_PATH = "./images"

class Bot:
    def __init__(self):
        
        img_list = os.listdir(IMAGE_PATH)
        self.image_paths = [Path(os.path.join(IMAGE_PATH, img_item)) for img_item in img_list if img_item.endswith(".jpg")]
        # self.show_index = random.sample(range(len(self.image_paths)), min(50, len(self.image_paths)))
        self.show_index = list(range(min(50, len(self.image_paths))))
        self.gallery_show_paths =  [self.image_paths[index] for index in self.show_index]

        with open(METAFILE_PATH,"r") as f:
            self.metadata = json.load(f)
    
    def add_gellary_image(self,user_state:dict,evt: gr.SelectData ):
        index = self.show_index[evt.index]
        print(f"\nselect No.{index} image", )
        return index, *self.add_image(user_state,type="index",index=index)
    
    def add_image(self, user_state:dict, 
                  index:int=0,
                  image_path:str = None, #path
                  type:Literal["random","image","index"] = "index",
                  ):

        
        if type == "image" and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")  
        elif type == "index" and index < len(self.image_paths):
            image_path =  self.image_paths[index] 
            image = Image.open(image_path).convert("RGB") 
        else:
            image_path = random.sample(self.image_paths, 1)[0]
            image = Image.open(image_path).convert("RGB")

        img_item = os.path.basename(image_path)
        print('\nupload an image: ',img_item)
        try:
            ann_path = self.metadata[img_item]
            with open(ann_path,"r") as f:
                ann = json.load(f)
        except Exception as e:
            print(e)
            return image, user_state
        

        data = {"origin_image":image,
                "path":image_path,
                "ann":ann["annotations"],
                "size":
                    {"width":
                        ann["image"]["width"],
                    "height":
                        ann["image"]["height"]
                    }
                }
        
        user_state.update(data)
        user_state["points"] = []
        return image, user_state
    
    def add_points(self, user_state:dict, evt: gr.SelectData):

        
        if user_state.get('origin_image', None) is None:
            img, user_state = self.add_image(user_state,type="random")
        else:
            img = user_state["origin_image"]

        # add points

        new_point = [evt.index[0], evt.index[1]]
        print("add point: ", new_point )

        if len(user_state.setdefault("points",[])) == 0 :
            user_state["points"].append(new_point)
        else:
            new_mask_points = [point for point in user_state["points"] 
                               if (new_point[0]- point[0])**2 + (new_point[1]- point[1])**2 > 225]
            if len(new_mask_points) == len(user_state["points"]):
                new_mask_points.append(new_point)
            user_state["points"] = new_mask_points

        if len(user_state["points"]) == 0:
            return None, img, user_state
        # find bbox
        candidate_bboxs = [bbox for bbox in user_state["ann"] if in_rectangle(bbox["box"],user_state["points"])]
        if len(candidate_bboxs) > 0:
            
            size  = [bbox["box"][2]*bbox["box"][3] for bbox in candidate_bboxs]

            final_bbox = candidate_bboxs[size.index(min(size))]
            x,y,w,h = tuple(final_bbox["box"])
            x1,y1,x2,y2 = int(x),int(y),int(x+w),int(y+h)
            user_state["final_ann"] = final_bbox
            label =  final_bbox["semantic_tag"][0]
            np_img = np.array(img)
            cv2_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            cv2.rectangle(cv2_image, (x1, y1), (x2,y2), (0, 255, 0), 4)
            cv2.putText(cv2_image,label, (int(x), int(y) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            new_image = self._blend_bbox(cv2_image_rgb, (x1,y1,x2,y2))
            new_image  = Image.fromarray(new_image)
            
        else:
            user_state["final_ann"] = {}
            new_image = img.copy()
            label = None
        # show image

        new_image = draw_points_to_image(new_image,user_state["points"])
        return label, new_image, user_state
        
    def save_img(self,user_stare:dict):
        img_path = user_stare.get("path",None)
        if img_path is not None:
            name = os.path.basename(img_path)
            new_path = os.path.join(SAVE_PATH,name)
            if not os.path.exists(new_path):
                shutil.copy(img_path, new_path)
                print("save image: ",name)
            else:
                print("The image path already exists.")
            return gr.update(value = "Saved!"), user_stare
        else:
            print("can't find image")
        return gr.update(value = "Save failed!"), user_stare

    def add_ann(self, user_state:dict):
        
        ann = user_state.get("final_ann",{})
        
        question = ann.get("question",[]).copy()
        question.append("Image caption")

        return gr.update(choices = question), user_state
    
    def update_answer(self,user_state:dict,evt: gr.SelectData):
        
        
        ann = user_state.get("final_ann",{})
        select_question = evt.value
        print("selected question:", select_question )

        if select_question in ann["question"]:
            answer = ann["answer"][min(evt.index,len(ann["answer"]))]
            print("selected answer:", answer )
            return  answer, user_state
        
        elif evt.index == len(ann["answer"]):
            return ann.get("caption",None), user_state
        
        else:
            print("selected answer: None")

            return None,user_state
      
    def update_all_answer(self, user_state:dict):
        ann = user_state.get("final_ann",{})
        question = ann.get("question",[]).copy()
        answer = ann.get("answer",[]).copy()
        caption = ann.get("caption", None)
        
        if caption is None:
            return None, user_state
        
        output = f"""Q1: {question[0]}
A1: {answer[0]}
        
Q2: {question[1]}
A2: {answer[1]}
        
Q3: {question[2]}
A3: {answer[2]}
        
Detailed Caption: {caption}
        """
        
        return output, user_state
        
    def _blend_bbox(self, img, bbox):
        x1,y1,x2,y2 = bbox
        mask = np.zeros_like(img)
        mask[y1:y2,x1:x2,:] = 255
        mask = mask.astype(np.uint8)
        mask[:,:,0] = 0
        mask[:,:,2] = 0
        new_img_arr = img * (1 - 1/3) + mask * 1/3
        new_img_arr = np.clip(new_img_arr, 0, 255).astype(np.uint8)
        # print(new_img_arr.shape)
        return new_img_arr

    def clear_points(self,user_state:dict):
        print("clear all points")
        
        user_state["points"] = []
        img = user_state.get("origin_image",None)
        return  img,user_state

    

         
class Seafoam(ThemeBase.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.emerald,
        secondary_hue=colors.blue,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_md,
        text_size=sizes.text_lg,
        font=(
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # body_background_fill="#D8E9EB",
            body_background_fill_dark="#111111",
            button_primary_background_fill="*primary_300",
            button_primary_background_fill_hover="*primary_200",
            button_primary_text_color="black",
            button_secondary_background_fill="*secondary_300",
            button_secondary_background_fill_hover="*secondary_200",
            border_color_primary="#0BB9BF",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="10px",
        )


css='''
#image_upload {align-items: center; max-width: 640px}
'''

def app(**kwargs):

    bot = Bot()

    with gr.Blocks(theme=Seafoam(), css=css) as demo:


        user_state = gr.State({})

        # gr.HTML(
        #     """
        #     <div align='center'> <h1>Dataset Browser For <a href="https://github.com/OpenGVLab/all-seeing">All-Seeing Project</h> </div>
        #     """,
        # )
        gr.HTML(
            """
            <div align='center'> <h1>The All-Seeing-1B(AS-1B) dataset Browser</h> </div>
            """,
        )
        # gr.HTML(
        #     """
        #     <div align='center'> <h1>Dataset Browser For <p align="center"><a href="https://github.com/OpenGVLab/all-seeing"><b>All-Seeing Project</b></p></h> </div>
        #     <p align="center">AS-1B contains over 1 billion regions annotated with semantic tags, question-answering pairs, and detailed captions. It covers a wide range of 3.5 million common and rare concepts in the real world, and has 132.2 billion tokens that describe the concepts and their attributes.</p>
        #     <p align="center"><a href="https://github.com/OpenGVLab/all-seeing"><b>All-Seeing Project</b></p>
        #     """,
        # )
        with gr.Row(visible=True) as user_interface:
            with gr.Column(scale=0.5, elem_id="text_input") as chat_part:
                with gr.Row(visible=True) as semantic_tag:
                    label = gr.Textbox(show_label=True,label="Semantic Tag",interactive=False)
                with gr.Row(visible=False) as question :  
                    question = gr.Dropdown([],label="Question",interactive=True)
                with gr.Row(visible=True) as answer:  
                    answer = gr.Textbox(show_label=True,label="Detailed Annotation",interactive=False, lines=12, max_lines=12)
                
                
            with gr.Column(elem_id="visual_input", scale=0.5) as img_part:
                # click_img = gr.AnnotatedImage(interactive=True, brush_radius=15, elem_id="image_upload",height=400)
                click_img = gr.Image(type="pil", interactive=False, brush_radius=15, elem_id="image_upload",height=392)

                with gr.Row(visible=False) as btn:  
                    select_img = gr.Slider(label="Image Index",minimum=0,maximum=len(bot.image_paths)-1,step=1,value=0)
                    # add_img_example = gr.Button("🖼️ Image Example", variant="primary")
                
                clear_btn = gr.Button(value="🗑️ Clear Points", variant="primary", elem_id="pick_btn")
                # save_btn = gr.Button(value="Save", variant="primary", elem_id="save_btn")

        with  gr.Row(visible=True) as gallery_row:
            gallery  = gr.Gallery(bot.gallery_show_paths ,label = "Image Gallery",columns = 8,allow_preview =False,height=385)

        # add_img_example.click(bot.add_image, [user_state], [click_img,user_state]).then(
        #     lambda: None, None, question).then(
        #     lambda: None, None, label)
        
        select_img.release(bot.add_image, [user_state,select_img], [click_img,user_state]).then(
            lambda: None, None, question).then(
            lambda: None, None, label)
        click_img.select(bot.add_points,[user_state,],[label, click_img, user_state]).then(
            bot.add_ann,[user_state],[question,user_state]).then(
            lambda: None, None, question).then(
            lambda: None, None, answer).then(
                bot.update_all_answer,[user_state],[answer,user_state]
            )

        question.select(bot.update_answer,[user_state],[answer,user_state])
            # pick_btn.click(lambda: gr.update(interactive=False), [], [clear_btn]).then(
            # ).then(
            #     bot.seg_image,[user_state],[click_img,user_state]).then(
            #     bot.add_image,[click_img, user_state], [ user_state]).then(
            # lambda: gr.update(interactive=True), [], [clear_btn])
            
        click_img.clear(lambda: {}, None, user_state).then(
            lambda: None, None, label).then(
            lambda: None, None, question).then(
            lambda: None, None, answer)
        
        clear_btn.click(bot.clear_points,[user_state],[click_img,user_state]).then(
            lambda: None, None, label).then(
            lambda: None, None, question).then(
            lambda: None, None, answer)

        gallery.select(bot.add_gellary_image,[user_state,],[select_img,click_img, user_state]).then(
            lambda: None, None, label).then(
            lambda: None, None, question).then(
            lambda: None, None, answer)

        # save_btn.click(bot.save_img,[user_state],[save_btn,user_state])


    demo.queue().launch(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=10019)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()

    # app(server_name="0.0.0.0", ssl_verify=False, server_port=args.port, share=args.share)
    app()
    # fire.Fire(app)