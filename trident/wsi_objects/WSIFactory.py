
import os
from typing import Optional, Literal, Union


from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
from trident.wsi_objects.ImageWSI import ImageWSI
from trident.wsi_objects.CuCIMWSI import CuCIMWSI
from trident.wsi_objects.ISyntaxWSI import ISyntaxWSI

WSIReaderType = Literal['openslide', 'image', 'cucim', 'isyntax']
ISYNTAX_EXTENSIONS = {'.isyntax'}
OPENSLIDE_EXTENSIONS = {'.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs'}
CUCIM_EXTENSIONS = {'.svs', '.tif', '.tiff'}

def load_wsi(
    slide_path: str,
    reader_type: Optional[WSIReaderType] = None,
    **kwargs
) -> Union[OpenSlideWSI, ImageWSI, CuCIMWSI, ISyntaxWSI]:
    """
    Load a whole-slide image (WSI) using the appropriate backend.

    By default, uses OpenSlideWSI for OpenSlide-supported file extensions,
    and ImageWSI for others. Users may override this behavior by explicitly
    specifying a reader using the `reader_type` argument.

    Parameters
    ----------
    slide_path : str
        Path to the whole-slide image.
    reader_type : {'openslide', 'image', 'cucim'}, optional
        Manually specify the WSI reader to use. If None (default), selection
        is automatic based on file extension.
    **kwargs : dict
        Additional keyword arguments passed to the WSI reader constructor.

    Returns
    -------
    Union[OpenSlideWSI, ImageWSI, CuCIMWSI]
        An instance of the appropriate WSI reader.

    Raises
    ------
    ValueError
        If `reader_type` is 'cucim' but the cucim package is not installed.
        Or if an unknown reader type is specified.
    """
    ext = os.path.splitext(slide_path)[1].lower()

    if reader_type == 'openslide':
        return OpenSlideWSI(slide_path=slide_path, **kwargs)

    elif reader_type == 'image':
        return ImageWSI(slide_path=slide_path, **kwargs)

    elif reader_type == 'cucim':
        if ext in CUCIM_EXTENSIONS:
            return CuCIMWSI(slide_path=slide_path, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}' for CuCIM. "
                f"Supported whole-slide image formats are: {', '.join(CUCIM_EXTENSIONS)}."
            )

    elif reader_type == 'isyntax':
        if ext in ISYNTAX_EXTENSIONS:
            max_workers = kwargs.get("max_workers", 1)
            kwargs.pop("max_workers")
            return ISyntaxWSI(
                slide_path=slide_path,
                max_workers=max_workers,
                **kwargs
            ) # isyntax SDK is not thread-safe
        else:
            raise ValueError(
                f"Unsupported file format '{ext}' for ISyntax. "
                f"Supported format is: .isyntax."
            )

    elif reader_type is None:
        if ext in ISYNTAX_EXTENSIONS:
            max_workers = kwargs.get("max_workers", 1)
            kwargs.pop("max_workers")
            return ISyntaxWSI(slide_path=slide_path, max_workers=max_workers, **kwargs)
        elif ext in OPENSLIDE_EXTENSIONS:
            return OpenSlideWSI(slide_path=slide_path, **kwargs)
        else:
            return ImageWSI(slide_path=slide_path, **kwargs)

    else:
        raise ValueError(f"Unknown reader_type: {reader_type}. Choose from 'openslide', 'image', 'cucim', or 'isyntax'.")
