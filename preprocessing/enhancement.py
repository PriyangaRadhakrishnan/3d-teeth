import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def enhance_dental_image(image_np: np.ndarray) -> np.ndarray:
    """
    Stage 2: Image Enhancement - Improves image quality and makes dental structures clearer
    using unsharp masking and contrast/sharpness enhancement without opencv dependency.
    """
    pil_img = Image.fromarray(image_np.astype(np.uint8))

    # Enhance contrast for enamel details
    enhancer = ImageEnhance.Contrast(pil_img)
    contrast_img = enhancer.enhance(1.25)

    # Sharpness enhancement for fine crown edges
    sharpener = ImageEnhance.Sharpness(contrast_img)
    sharp_img = sharpener.enhance(1.5)

    # Unsharp mask filter
    unsharp_img = sharp_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    return np.array(unsharp_img, dtype=np.uint8)
