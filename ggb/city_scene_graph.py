import os, sys
import numpy as np
from openai import OpenAI
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from gsamllavanav.maps.landmark_map import LandmarkMap
from gsamllavanav.defaultpaths import GDINO_CHECKPOINT_PATH, GDINO_CONFIG_PATH, SAM_CHECKPOINT_PATH, MOBILE_SAM_CHECKPOINT_PATH, GSAM_MAPS_DIR
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap

class LandmarkNode():
    def __init__(self, Landmark_caption):
        self.Landmark_caption = Landmark_caption
        self.nodes = set() # no replicated elements

class edges():
    def __init__(self):
        self.attr = None
        self.nodes = set()

class SceneGraph():
    def __init__(self, agent, llm_name='GPT', map_name='birmingham_block_1', map_shape=(240, 240), pixels_per_meter=0.5853658536585366, landmark_names=['Leslie Road', 'Wellington Road']):
        self.agent = agent
        self.nodes = set()
        self.subgraphs = set()
        self.Landmarkmap = LandmarkMap(map_name=map_name,map_shape=map_shape,pixels_per_meter=pixels_per_meter,landmark_names=landmark_names)
        self.init_landmark_nodes = self.init_landmark_nodes()
    
    def init_landmark_nodes(self):
        # TODO FIND HOW TO DESCRIBE A LANDMARK NODE, CAPTION OR WHAT?
        landmark_nodes = []
        for landmark_caption in self.agent.rooms:
            landmark_node = LandmarkNode(landmark_caption)
            landmark_nodes.append(landmark_node)
        return landmark_nodes
    
    def get_sam_mask_generator(self, variant:str, device) -> SamAutomaticMaskGenerator:
        if variant == 'sam':
            sam = sam_model_registry["vit_h"](SAM_CHECKPOINT_PATH)
            sam.to(device=device)
            mask_generator = SamAutomaticMaskGenerator(sam)
            return mask_generator
        elif variant == "MobileSAM":
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    def get_sam_segmentation_dense(
            self, variant:str, model, image: np.ndarray
    ) -> tuple:
        masks = model.generate(image)
        for mask in masks:
            bbox = mask['bbox']
            segmentation = mask['segmentation']
            conf = mask["predicted_iou"]
    def segment2d(self, image_rgb):
        print(' segment2d...')
        m,xy,conf = self.get_sam_segmentation_dense(self.sam_variant)

    def llm(self, prompt):
        if self.llm_name == 'GPT':
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                chat_completion = client.chat.completions.create(  # added by someone
                    model="gpt-3.5-turbo",
                    # model="gpt-4",  # gpt-4
                    messages=[{"role": "user", "content": prompt}],
                    # timeout=10,  # Timeout in seconds
                )
                return chat_completion.choices[0].message.content
            except:
                return ''
    def clear_line(self):
        sys.stdout.write('\033[F')
        sys.stdout.write('\033[J')
        sys.stdout.flush()
    
if __name__ == '__main__':
    
    ggb = SceneGraph()