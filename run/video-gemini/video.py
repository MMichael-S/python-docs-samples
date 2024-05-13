# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START cloudrun_imageproc_handler_setup]
import os
import json
import tempfile

from google.cloud import storage
# from wand.image import Image

import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

storage_client = storage.Client()
vertexai.init(project="shifei-demo-1", location="us-central1")

model = GenerativeModel(
    "gemini-1.5-pro-preview-0409",
)

# [END cloudrun_imageproc_handler_setup]

def is_json(myjson):
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True

def video_gemini_analysis(data):
    """
       Args:
        data: Pub/Sub message data
    file_data = data
    """

    file_name = file_data["name"]
    bucket_name = file_data["bucket"]

    blob_uri = f"gs://{bucket_name}/{file_name}"

    prompt_text = """将视频内容以每 5 秒为单位，对视频内容进行识别，包括：
    视频中的物体
    人物的年龄、肤色、性别、行为、表情

    如果您对任何信息不确定，请不要编造。 

    用中文以 JSON 格式返回结果，不要将 json 结果包装在 JSON 标记中

    参考JSON格式：
    {
      \"chapters\":[
       {
         \"timecode\":\"00:00\",
         \"chapterSummary\":\"\",
         \"objects\":[
          \"沙发\",
          \"笔记本电脑\",
          \"壁炉\",
          \"火焰\"
         ],
         \"people\":[
          {
            \"age\":\"成年\",
            \"skin_color\":\"黑人\",
            \"gender\":\"女\",
            \"behavior\":\"看着笔记本电脑\",
            \"emotion\":\"微笑\"
          },
          {
            \"age\":\"儿童\",
            \"skin_color\":\"黑人\",
            \"gender\":\"女\",
            \"behavior\":\"使用笔记本电脑\",
            \"emotion\":\"微笑\"
          }
         ]
       },
      ]
    }"""

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0.1,
        "top_p": 1,
        # "responseMimeType": "text/plain",
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    # print(blob_uri)
    
    video1 = Part.from_uri(
        mime_type="video/mp4",
        uri=blob_uri)

    responses = model.generate_content(
        [video1, prompt_text],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    )

    # print(responses)
    # print("Response:\n",json.loads(responses.text))
    
    if is_json(responses.text) == True:
        print("Response is JSON\n")
        print("Response:\n",json.loads(responses.text))
        
        return(json.loads(responses.text))
    else:
        print("Response Not JSON Error")
