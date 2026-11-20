import PIL
from PIL import Image, ImageDraw

def concat_image(images: list, canvas_size: list, labels: list = None, font_size: float = 0) -> PIL.Image:
    """
    Method:
        dirname = "data/images"
        images_names = ["image_1_ori.jpg", "image_2_ori.jpg", "image_3_ori.jpg"]
        images_names = [os.path.join(dirname, name) for name in images_names]
        images = [Image.open(n) for n in images_names]
        labels =  ["A", "B", "C"]
        canvas_size, font_size = (20, 20), 60
        image_concat = concat_image(images, canvas_size, labels, font_size)
    """

    image_mode = "RGB"
    margin, padding = canvas_size
    min_height = min(im.height for im in images)
    images_resize = [im.resize((int(im.width * min_height / im.height), min_height), resample=PIL.Image.Resampling.LANCZOS) for im in images]
    total_width = sum(im.width for im in images_resize)

    interval = padding * (len(images_resize) - 1)

    dst_width = total_width + margin * 2 + interval
    dst_height = min_height + margin * 2 + font_size + margin
    dst = Image.new(image_mode, (dst_width, dst_height), color="white")

    draw = ImageDraw.Draw(dst)

    pos_x, pos_y = margin, margin

    font_kwargs = {"font_size": font_size, "align": "center"}

    for i, im in enumerate(images_resize):
        box = (pos_x, pos_y, pos_x + im.width, pos_y + im.height)
        dst.paste(im.convert(image_mode), box)

        if labels is not None and i < len(labels):
            label_size = draw.textbbox((0, 0), labels[i], **font_kwargs)
            label_x = pos_x + (im.width - label_size[0]) // 2
            label_y = pos_y + im.height + (font_size - label_size[1]) // 2
            draw.text((label_x, label_y), labels[i], fill="black", **font_kwargs)

        pos_x += im.width + padding
    return dst
