import io
import base64



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

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")