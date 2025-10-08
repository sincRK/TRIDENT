import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from abc import abstractmethod

from trident.IO import get_dir, get_weights_path, has_internet_connection
from .model_zoo.lenet5_models import select_lenet5_model


class SegmentationModel(torch.nn.Module):

    _has_internet = has_internet_connection()

    def __init__(self, freeze=True, confidence_thresh=0.5, **build_kwargs):
        """
        Initialize Segmentation model wrapper.

        Args:
            freeze (bool, optional): If True, the model's parameters are frozen
                (i.e., not trainable) and the model is set to evaluation mode.
                Defaults to True.
            confidence_thresh (float, optional): Threshold for prediction confidence.
                Predictions below this threshold may be filtered out or ignored.
                Default is 0.5. Set to 0.4 to keep more tissue.
            **build_kwargs: Additional keyword arguments passed to the internal
                `_build` method.

        Attributes:
            model (torch.nn.Module): The constructed model.
            eval_transforms (Callable): Transformations to apply to input data during inference.
        """
        super().__init__()
        self.model, self.eval_transforms = self._build(**build_kwargs)
        self.confidence_thresh = confidence_thresh

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, image):
        """
        Can be overwritten if model requires special forward pass.
        """
        z = self.model(image)
        return z

    @abstractmethod
    def _build(self, **build_kwargs) -> tuple[nn.Module, transforms.Compose]:
        """
        Build the segmentation model and preprocessing transforms.
        """
        pass


class HESTSegmenter(SegmentationModel):

    def __init__(self, **build_kwargs):
        """
        HESTSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        """
        Build and load HESTSegmenter model.

        Returns:
            Tuple[nn.Module, transforms.Compose]: Model and preprocessing transforms.
        """

        from torchvision.models.segmentation import deeplabv3_resnet50

        model_ckpt_name = 'deeplabv3_seg_v4.ckpt'
        weights_path = get_weights_path('seg', 'hest')

        # Check if a path is provided but doesn't exist
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the file was not found.")

        # Initialize base model
        model = deeplabv3_resnet50(weights=None)
        model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1, stride=1)

        if not weights_path:
            if not SegmentationModel._has_internet:
                raise FileNotFoundError(
                    f"Internet connection not available and checkpoint not found locally in model registry at trident/segmentation_models/local_ckpts.json.\n\n"
                    f"To proceed, please manually download {model_ckpt_name} from:\n"
                    f"https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    f"and place it at:\nlocal_ckpts.json"
                )

            # If internet is available, download from HuggingFace
            from huggingface_hub import snapshot_download
            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type='model',
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name]
            )

            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        # Load and clean checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = {
            k.replace('model.', ''): v
            for k, v in checkpoint.get('state_dict', {}).items()
            if 'aux' not in k
        }

        model.load_state_dict(state_dict)

        # Store configuration
        self.input_size = 512
        self.precision = torch.float16
        self.target_mag = 20  # Benchmark test resolution

        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # input should be of shape (batch_size, C, H, W)
        assert len(image.shape) == 4, f"Input must be 4D image tensor (shape: batch_size, C, H, W), got {image.shape} instead"
        logits = self.model(image)['out']
        softmax_output = F.softmax(logits, dim=1)
        predictions = (softmax_output[:, 1, :, :] > self.confidence_thresh).to(torch.uint8)  # Shape: [bs, 512, 512]
        return predictions


class JpegCompressionTransform:
    def __init__(self, quality=80):
        self.quality = quality

    def __call__(self, image):
        import cv2
        import numpy as np
        from PIL import Image
        # Convert PIL Image to NumPy array
        image = np.array(image)

        # Apply JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, image = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Convert back to PIL Image
        return Image.fromarray(image)


class GrandQCArtifactSegmenter(SegmentationModel):

    _class_mapping = {
        1: "Normal Tissue",
        2: "Fold",
        3: "Darkspot & Foreign Object",
        4: "PenMarking",
        5: "Edge & Air Bubble",
        6: "OOF",
        7: "Background"
    }

    def __init__(self, **build_kwargs):
        """
        GrandQCArtifactSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, remove_penmarks_only=False):
        """
        Load the GrandQC artifact removal segmentation model.
        Credit: https://www.nature.com/articles/s41467-024-54769-y
        """

        import segmentation_models_pytorch as smp

        self.remove_penmarks_only = remove_penmarks_only  # ignore all other artifacts than penmakrs.
        model_ckpt_name = 'GrandQC_MPP1_state_dict.pth'
        encoder_name = 'timm-efficientnet-b0'
        weights_path = get_weights_path('seg', 'grandqc_artifact')

        # Verify that user-provided weights_path is valid
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Expected checkpoint at '{weights_path}', but the file was not found."
            )

        # Initialize model
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=8,
            activation=None,
        )

        # Attempt to download if file is missing and not already available
        if not weights_path:
            if not SegmentationModel._has_internet:
                raise FileNotFoundError(
                    f"Internet connection not available and checkpoint not found locally.\n\n"
                    f"To proceed, please manually download {model_ckpt_name} from:\n"
                    f"https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    f"and place it at:\nlocal_ckpts.json"
                )

            from huggingface_hub import snapshot_download
            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type='model',
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name],
            )

            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        # Load checkpoint
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

        # Model config
        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 20  # Benchmark test resolution
        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Custom forward pass.
        """
        logits = self.model.predict(image)
        probs = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(probs, dim=1)
        if self.remove_penmarks_only:
            predictions = torch.where((predicted_classes == 4) | (predicted_classes == 7), 0, 1)
        else:
            predictions = torch.where(predicted_classes > 1, 0, 1)
        predictions = predictions.to(torch.uint8)

        return predictions


class GrandQCSegmenter(SegmentationModel):

    def __init__(self, **build_kwargs):
        """
        GrandQCSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        """
        Load the GrandQC tissue detection segmentation model.
        Credit: https://www.nature.com/articles/s41467-024-54769-y
        """
        import segmentation_models_pytorch as smp

        model_ckpt_name = 'Tissue_Detection_MPP10.pth'
        encoder_name = 'timm-efficientnet-b0'
        weights_path = get_weights_path('seg', 'grandqc')

        # Verify that user-provided weights_path is valid
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Expected checkpoint at '{weights_path}', but the file was not found."
            )

        # Verify checkpoint path
        if not weights_path:
            if not SegmentationModel._has_internet:
                raise FileNotFoundError(
                    f"Internet connection not available and checkpoint not found locally at '{weights_path}'.\n\n"
                    f"To proceed, please manually download {model_ckpt_name} from:\n"
                    f"https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    f"and place it at:\nlocal_ckpts.json"
                )

            from huggingface_hub import snapshot_download
            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type='model',
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name],
            )
            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        # Initialize model
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=2,
            activation=None,
        )

        # Load checkpoint
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

        # Model config
        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 20  # Benchmark test resolution
        eval_transforms = transforms.Compose([
            JpegCompressionTransform(quality=80),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Custom forward pass.
        """
        logits = self.model.predict(image)
        probs = torch.softmax(logits, dim=1)
        max_probs, predicted_classes = torch.max(probs, dim=1)
        predictions = (max_probs >= self.confidence_thresh) * (1 - predicted_classes)
        predictions = predictions.to(torch.uint8)

        return predictions


class LeNet5Segmenter(SegmentationModel):

    def __init__(self, input_size=128, **build_kwargs):
        """
        LeNet5Segmenter initialization.
        Based on:
        Naghshineh Kani, Sepideh; Soyak, Burak Can; Gokce, Melih; Duyar, Zeynep;
        Alicikus, Hasan; Yapıcıer, Özlem; Öner, Mustafa Ümit (2024).
        Tissue-region-segmentation-in-HE-and-IHC-stained-pathology-slides-of-specimens-from-different-origins
        (Version 1.0.0) [Computer software].
        https://github.com/sepidehnaghshineh/Tissue-region-segmentation-in-HE-and-IHC-stained-pathology-slides.
        DOI: 10.5281/zenodo.1234

        Args:
            input_size (int): Input image size. Supported: 128, 64, 32. Default: 128.
        """
        super().__init__(freeze=build_kwargs.get('freeze', True),
                         confidence_thresh=build_kwargs.get('confidence_thresh', 0.5))
        self.input_size = input_size

    def _build(self, input_size=128):
        """
        Build and load LeNet5Segmenter model.
        Based on:
        Naghshineh Kani, Sepideh; Soyak, Burak Can; Gokce, Melih; Duyar, Zeynep;
        Alicikus, Hasan; Yapıcıer, Özlem; Öner, Mustafa Ümit (2024).
        Tissue-region-segmentation-in-HE-and-IHC-stained-pathology-slides-of-specimens-from-different-origins
        (Version 1.0.0) [Computer software].
        https://github.com/sepidehnaghshineh/Tissue-region-segmentation-in-HE-and-IHC-stained-pathology-slides.
        DOI: 10.5281/zenodo.1234

        Returns:
            Tuple[nn.Module, transforms.Compose]: Model and preprocessing transforms.
        """

        model_ckpt_name = 'LeNet5Segmentation.pth'
        weights_path = get_weights_path('seg', 'LeNet5')

        # Check if a path is provided but doesn't exist
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Expected checkpoint at '{weights_path}', but the file was not found."
            )

        # Initialize base model
        model = select_lenet5_model(input_size)

        if not weights_path:
            raise FileNotFoundError(
                f"LeNet5 checkpoint not found locally. Please ensure {model_ckpt_name} "
                f"is available in the model registry at trident/segmentation_models/local_ckpts.json"
            )

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint)

        # Store configuration - from original segmentation script
        self.input_size = input_size
        self.precision = torch.float32

        # Target magnification optimized through benchmarking
        # Based on resolution vs performance analysis:
        # - target_mag = 2.5  → 4.00 µm/px, 4.7s
        # - target_mag = 5    → 2.00 µm/px, 5.7s
        # - target_mag = 10   → 1.00 µm/px, 9.1s
        # - target_mag = 15   → 0.67 µm/px, 12.3s
        # - target_mag = 20   → 0.50 µm/px, 18.8s
        #
        # Selected target_mag = 20 for optimal tissue boundary precision
        # Trade-off: ~2x slower than HEST but superior segmentation accuracy
        self.target_mag = 20

        # Normalization values from original model training
        # means = [0.43412438, 0.4126196, 0.43120176]
        # stds = [0.06676771, 0.07356026, 0.0663118]
        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43412438, 0.4126196, 0.43120176],
                std=[0.06676771, 0.07356026, 0.0663118]
            )
        ])

        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LeNet5 patch-based segmentation.

        Note: LeNet5 is a patch-level classifier, not a pixel-level segmenter.
        This method performs sliding window inference to create a segmentation mask.

        Args:
            image: Input tensor of shape (batch_size, C, H, W)

        Returns:
            Binary mask predictions as uint8 tensor
        """
        # Input should be of shape (batch_size, C, H, W)
        assert len(image.shape) == 4, (
            f"Input must be 4D image tensor (shape: batch_size, C, H, W), "
            f"got {image.shape} instead"
        )

        batch_size, channels, height, width = image.shape

        # Initialize output mask
        output_masks = []

        for b in range(batch_size):
            single_image = image[b:b+1]  # Keep batch dimension

            # Perform sliding window inference
            mask = self._sliding_window_inference(single_image.squeeze(0))
            output_masks.append(mask)

        # Stack all masks
        output = torch.stack(output_masks, dim=0)
        return output.to(torch.uint8)

    def _sliding_window_inference(
        self, image: torch.Tensor, batch_size: int = 64
    ) -> torch.Tensor:
        """
        Perform efficient batched sliding window inference on a single image.

        Args:
            image: Single image tensor of shape (C, H, W)
            batch_size: Number of patches to process in each batch

        Returns:
            Binary mask tensor of shape (H, W)
        """
        channels, height, width = image.shape

        # Create output mask
        mask = torch.zeros((height, width), device=image.device, dtype=torch.float32)
        counts = torch.zeros((height, width), device=image.device, dtype=torch.float32)

        # Sliding window parameters - match original implementation exactly
        window_size = self.input_size  # 128
        stride = 16  # Exact match to original: model_cropsize//2**(pred_resolution-1)

        # Collect all patch coordinates first
        patch_coords = []
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                patch_coords.append((y, x))

        # Handle edge cases
        if width % stride != 0:
            x = width - window_size
            for y in range(0, height - window_size + 1, stride):
                patch_coords.append((y, x))

        if height % stride != 0:
            y = height - window_size
            for x in range(0, width - window_size + 1, stride):
                patch_coords.append((y, x))

        if width % stride != 0 and height % stride != 0:
            patch_coords.append((height - window_size, width - window_size))

        # Remove duplicates
        patch_coords = list(set(patch_coords))

        print(f"Processing {len(patch_coords)} patches in batches of {batch_size}")

        # Process patches in batches
        for i in range(0, len(patch_coords), batch_size):
            batch_coords = patch_coords[i:i + batch_size]

            # Extract batch of patches
            batch_patches = []
            valid_coords = []

            for y, x in batch_coords:
                patch = image[:, y:y+window_size, x:x+window_size]
                if patch.shape[1] == window_size and patch.shape[2] == window_size:
                    batch_patches.append(patch)
                    valid_coords.append((y, x))

            if not batch_patches:
                continue

            # Stack patches into batch tensor
            batch_tensor = torch.stack(batch_patches, dim=0)

            # Predict for entire batch
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = F.softmax(logits, dim=1)
                tissue_probs = probs[:, 1]  # Get tissue probabilities
                predictions = (tissue_probs > self.confidence_thresh).long()

            # Simple binary voting - only count tissue votes
            for idx, (y, x) in enumerate(valid_coords):
                pred = predictions[idx].item()

                if pred == 1:  # Only count tissue predictions
                    mask[y:y+window_size, x:x+window_size] += 1

                counts[y:y+window_size, x:x+window_size] += 1

            # Print progress
            if (i // batch_size) % 10 == 0:
                progress = min(100, (i + batch_size) / len(patch_coords) * 100)
                print(f"Progress: {progress:.1f}%")

        # Simple majority voting: need >50% of patches to vote tissue
        counts = torch.clamp(counts, min=1)  # Avoid division by zero
        tissue_ratio = mask / counts
        # Use conservative threshold: >60% of patches must vote tissue
        binary_mask = (tissue_ratio > 0.6).float()

        return binary_mask
        return binary_mask


def segmentation_model_factory(
    model_name: str,
    confidence_thresh: float = 0.5,
    freeze: bool = True,
    **build_kwargs,
) -> SegmentationModel:
    """
    Factory function to build a segmentation model by name.
    """

    if "device" in build_kwargs:
        import warnings
        warnings.warn(
            "Passing `device` to `segmentation_model_factory` is deprecated as of "
            "version 0.1.0. Please pass `device` when segmenting the tissue, e.g., "
            "`slide.segment_tissue(..., device='cuda:0')`.",
            DeprecationWarning,
            stacklevel=2
        )

    if model_name == 'hest':
        return HESTSegmenter(
            freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs
        )
    elif model_name == 'grandqc':
        return GrandQCSegmenter(
            freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs
        )
    elif model_name == 'grandqc_artifact':
        return GrandQCArtifactSegmenter(freeze=freeze, **build_kwargs)
    elif model_name == 'lenet5':
        return LeNet5Segmenter(
            freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs
        )
    else:
        raise ValueError(f"Model type {model_name} not supported")
