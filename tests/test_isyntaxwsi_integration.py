import os
import unittest
import torch
from trident import load_wsi
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

"""
Integration test for ISyntaxWSI: end-to-end pipeline on iSyntax files.
This test mirrors the OpenSlideWSI integration test, but for Philips iSyntax format.
"""


class TestISyntaxWSIIntegration(unittest.TestCase):
    """Integration test for ISyntaxWSI: end-to-end pipeline on iSyntax files."""

    TEST_ISYNTAX_FILENAMES = [
        "f7da653f-6002-303f-8fb7-fcb939a6414b.isyntax"
    ]
    TEST_OUTPUT_DIR = "test_isyntax_slide_processing/"
    TEST_PATCH_ENCODER = "uni_v2"
    TEST_MAG = 20
    TEST_PATCH_SIZE = 256
    TEST_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    TEST_ISYNTAX_DIR = "/data/floriansauter/05jun2023/001/"


    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.TEST_OUTPUT_DIR, exist_ok=True)

    def test_integration(self):
        """Run the full pipeline on a test iSyntax slide and check outputs."""

        for slide_filename in self.TEST_ISYNTAX_FILENAMES:
            with self.subTest(slide=slide_filename):
                slide_path = os.path.join(self.TEST_ISYNTAX_DIR, slide_filename)
                slide = load_wsi(
                    slide_path=slide_path,
                    reader_type='isyntax',
<<<<<<< HEAD
                    lazy_init=False,
                    max_workers=1
=======
                    lazy_init=False
>>>>>>> test basic loading and integration
                )

                # Step 1: Tissue segmentation
                segmentation_model = segmentation_model_factory("hest")
                slide.segment_tissue(
                    segmentation_model=segmentation_model,
                    target_mag=10,
                    job_dir=self.TEST_OUTPUT_DIR,
                    device=self.TEST_DEVICE
                )

                # Step 2: Tissue coordinate extraction
                coords_path = slide.extract_tissue_coords(
                    target_mag=self.TEST_MAG,
                    patch_size=self.TEST_PATCH_SIZE,
                    save_coords=self.TEST_OUTPUT_DIR
                )

                # Step 3: Visualization
                viz_coords_path = slide.visualize_coords(
                    coords_path=coords_path,
                    save_patch_viz=os.path.join(self.TEST_OUTPUT_DIR, "visualization")
                )

                # Step 4: Feature extraction
                encoder = encoder_factory(self.TEST_PATCH_ENCODER)
                encoder.eval()
                encoder.to(self.TEST_DEVICE)
                features_dir = os.path.join(
                    self.TEST_OUTPUT_DIR,
                    f"features_{self.TEST_PATCH_ENCODER}"
                )
                slide.extract_patch_features(
                    patch_encoder=encoder,
                    coords_path=coords_path,
                    save_features=features_dir,
                    device=self.TEST_DEVICE
                )

                # Verify outputs
                self.assertTrue(
                    os.path.exists(
                        os.path.join(self.TEST_OUTPUT_DIR, "contours_geojson")
                    ),
                    "GDF contours were not saved."
                )
                self.assertTrue(
                    os.path.exists(
                        os.path.join(self.TEST_OUTPUT_DIR, "contours")
                    ),
                    "Contours were not saved."
                )
                self.assertTrue(
                    os.path.exists(coords_path),
                    "Tissue coordinates file was not saved."
                )
                self.assertTrue(
                    os.path.exists(viz_coords_path),
                    "Visualization file was not saved."
                )
                self.assertTrue(
                    os.path.exists(features_dir),
                    "Feature extraction results were not saved."
                )

        # Check output file counts
        expected_file_count = len(self.TEST_ISYNTAX_FILENAMES)
        output_dirs = [
            "visualization",
            "thumbnails",
            "patches",
            "contours",
            "contours_geojson",
            f"features_{self.TEST_PATCH_ENCODER}"
        ]
        for output_dir in output_dirs:
            dir_path = os.path.join(self.TEST_OUTPUT_DIR, output_dir)
            self.assertTrue(
                os.path.exists(dir_path),
                f"Directory '{output_dir}' does not exist."
            )
            files = [
                f for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
            ]
            self.assertEqual(
                len(files), expected_file_count,
                (
                    f"Expected {expected_file_count} files in '{output_dir}', "
                    f"but found {len(files)}."
                )
            )


if __name__ == "__main__":
    unittest.main()
