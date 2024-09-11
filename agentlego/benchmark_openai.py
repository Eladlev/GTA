from typing import List, Optional
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from agentlego.tools import BaseTool
from agentlego.types import ImageIO, Annotated, Info
import yaml

import base64
import requests


LLM_ENV = yaml.safe_load(open('llm_env.yml', 'r'))

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


class GPTInferencer:

    def __init__(self,
                 model='gpt-4o-mini',
                 revision='f57cfbd358cb56b710d963669ad1bcfb44cdcdd8',
                 fp16=False,
                 device=None):

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_ENV['OPENAI_API_KEY']}"
        }
        self.model = model

    def __call__(self, image: ImageIO, text: str):
        # Getting the base64 string
        base64_image = encode_image(image.to_path())
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        response = response.json()
        return response['choices']#[0]['message']['content'][0]['text']



class ImageDescription(BaseTool):
    default_desc = ('A useful tool that returns a brief '
                    'description of the input image.')

    def __init__(self,
                 model: str = 'gpt-4o-mini',
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device

    def setup(self):
        self._inferencer = GPTInferencer(self.model)

    def apply(self, image: ImageIO) -> str:
        image = image.to_array()[:, :, ::-1]
        return self._inferencer(image, 'Describe the image in detail')[0]['pred_answer']


class CountGivenObject(BaseTool):
    default_desc = 'The tool can count the number of a certain object in the image.'

    def __init__(self,
                 model: str = 'gpt-4o-mini',
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device

    def setup(self):
        self._inferencer = GPTInferencer(self.model)

    def apply(
        self,
        image: ImageIO,
        text: Annotated[str, Info('The object description in English.')],
        bbox: Annotated[Optional[str],
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')] = None,
    ) -> int:
        import re
        if bbox is None:
            image = image.to_array()[:, :, ::-1]
        else:
            from agentlego.utils import parse_multi_float
            x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
            image = image.to_array()[y1:y2, x1:x2, ::-1]
        res = self._inferencer(image, f'How many {text} are in the image? Reply a digit')[0]['pred_answer']
        res = re.findall(r'\d+', res)
        if len(res) > 0:
            return int(res[0])
        else:
            return 0


class RegionAttributeDescription(BaseTool):
    default_desc = 'Describe the attribute of a region of the input image.'

    def __init__(self,
                 model: str = 'gpt-4o-mini',
                 device: str = 'cuda',
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.device = device

    def setup(self):
        self._inferencer = GPTInferencer(self.model)

    def apply(
        self,
        image: ImageIO,
        bbox: Annotated[str,
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
        attribute: Annotated[str, Info('The attribute to describe')],
    ) -> str:
        from agentlego.utils import parse_multi_float
        x1, y1, x2, y2 = (int(item) for item in parse_multi_float(bbox))
        cropped_image = image.to_array()[y1:y2, x1:x2, ::-1]
        return self._inferencer(cropped_image, f'Describe {attribute} on the image in detail')[0]['pred_answer']
