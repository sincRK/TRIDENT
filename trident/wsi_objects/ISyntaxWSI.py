
import numpy as np
from isyntax import ISyntax
from .WSI import WSI
from typing import Optional, Tuple, Union
from PIL import Image


class ISyntaxWSI(WSI):
    """
    WSI backend for Philips iSyntax (.isyntax) files using pyisyntax.
    Inherits from the base WSI class and implements required methods for iSyntax support.
    """

    def _lazy_initialize(self) -> None:
        """
        Initialize iSyntax backend and set all required WSI attributes.
        """
        super()._lazy_initialize()
        if not self.lazy_init:
            try:
                self.img = ISyntax.open(self.slide_path)
                self.dimensions = self.img.dimensions
                self.width, self.height = self.dimensions
                self.level_count = self.img.level_count
                self.level_downsamples = self.img.level_downsamples
                self.level_dimensions = self.img.level_dimensions
                self.properties = {
                    "mpp_x": self.img.mpp_x,
                    "mpp_y": self.img.mpp_y,
                    "barcode": getattr(self.img, "barcode", None)
                }
                if self.mpp is None:
                    self.mpp = self._fetch_mpp(self.custom_mpp_keys)
                self.mag = self._fetch_magnification(self.custom_mpp_keys)
                self.lazy_init = True
            except Exception as e:
                raise RuntimeError(f"Failed to initialize WSI with ISyntax: {e}") from e

    def _fetch_mpp(self, custom_mpp_keys: Optional[list] = None) -> float:
        """
        Retrieve microns per pixel (MPP) from ISyntax metadata.

        Args:
            custom_mpp_keys (list, optional): Additional keys to check for MPP.

        Returns:
            float: MPP value in microns per pixel.

        Raises:
            ValueError: If MPP cannot be determined from metadata.
        """
        # ISyntax exposes mpp_x and mpp_y directly
        if hasattr(self.img, "mpp_x") and self.img.mpp_x:
            return float(self.img.mpp_x)
        if custom_mpp_keys:
            for key in custom_mpp_keys:
                if key in self.properties:
                    try:
                        return float(self.properties[key])
                    except Exception:
                        continue
        raise ValueError(
            f"Unable to extract MPP from ISyntax metadata: '{self.slide_path}'"
        )

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: str = 'numpy'
    ) -> Union[np.ndarray, Image.Image]:
        """
        Read a region from the slide at the given location, level, and size.
        Returns a numpy array (H, W, 3) in RGB or PIL Image if read_as='pil'.
        Default behaviour is different than openslide in the following:
        openslide:
        - location always as mag level 0 indepentent of "level" param
        - size is based on pixel at level

        pyisyntax:
        - location scales with the level param
        - size is based on pixel at level

        To make behaviour close to openslide we have to scale the location
        """
        scale = 2 ** (-level)
        x, y = int(location[0] * scale), int(location[1] * scale)
        width, height = size
        arr = self.img.read_region(x, y, width, height, level)
        arr = arr[..., :3]  # RGB
        if read_as == 'pil':
            from PIL import Image
            return Image.fromarray(arr)
        return arr

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the dimensions (width, height) of the WSI.
        """
        return self.img.dimensions

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        """
        Return a thumbnail as a RGB PIL.Image.Image.
        """
        thumb_level = self.level_count - 1
        arr = self.img.read_region(
            0, 0,
            self.level_dimensions[thumb_level][0],
            self.level_dimensions[thumb_level][1],
            thumb_level
        )
        from PIL import Image
        img = Image.fromarray(arr[..., :3])
        img = img.resize(size)
        return img.convert("RGB")  # Ensure RGB format
