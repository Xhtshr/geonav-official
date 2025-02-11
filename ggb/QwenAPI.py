from openai import OpenAI
import os
import io
import base64

from PIL import Image

rgb = Image.open("Qwen/car_grounding.jpg")
def encode_image_from_pil(image):
    """
    将PIL.Image对象编码为Base64格式的字符串

    参数:
    image (PIL.Image): PIL.Image对象

    返回:
    str: Base64编码后的图像字符串
    """
    # 创建一个字节流对象
    image_byte_array = io.BytesIO()
    # 将PIL.Image对象保存到字节流中，格式为PNG
    image.save(image_byte_array, format='PNG')
    # 获取字节流中的数据
    image_byte_array = image_byte_array.getvalue()
    # 使用Base64进行编码，并解码为UTF-8格式的字符串
    return base64.b64encode(image_byte_array).decode("utf-8")

# 示例用法
base64_string = encode_image_from_pil(rgb)

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# base64_image = encode_image("test.png")
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-f0de3487904a4a11950ba707623cdbab",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-vl-max-latest",
    messages=[
    	{
    	    "role": "system",
            "content": [{"type":"text","text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # 需要注意，传入BASE64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/png;base64,{base64_string}"}, 
                },
                {"type": "text", "text": """detect the cars in the image, return bounding boxes for all of them, and the  using the following format: [{
        "object": "object_name",
        "bboxes": [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...],
        "features":['feature1', 'feature2',]
     }, ...]"""},
            ],
        }
    ],
)
print(completion.choices[0].message.content)