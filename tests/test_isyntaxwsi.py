"""Unit tests for verifying the ISyntax WSI (Whole Slide Image) interface."""

import os
import unittest


from trident.wsi_objects.ISyntaxWSI import ISyntaxWSI

<<<<<<< HEAD
PATH_TO_ISYNTAX = "/data/floriansauter/05jun2023/001/f7da653f-6002-303f-8fb7-fcb939a6414b.isyntax"
=======
PATH_TO_ISYNTAX = "/home/tobechanged/mirrored_folder/minidata/blau/blau_batch0_14oct2022_case033_sample1.isyntax"
>>>>>>> test basic loading and integration


class TestISyntaxWSI(unittest.TestCase):
    """Test ISyntaxWSI integration with TRIDENT WSI interface."""

    def setUp(self):
        self.assertTrue(os.path.isfile(PATH_TO_ISYNTAX), f"File does not exist: {PATH_TO_ISYNTAX}")
        self.wsi = ISyntaxWSI(slide_path=PATH_TO_ISYNTAX, lazy_init=False)

    def tearDown(self):
        self.wsi.release()

    def test_metadata_and_properties(self):
        self.assertEqual(self.wsi.width, self.wsi.img.dimensions[0])
        self.assertEqual(self.wsi.height, self.wsi.img.dimensions[1])
        self.assertGreater(self.wsi.level_count, 0)
        self.assertIsInstance(self.wsi.level_downsamples, list)
        self.assertIsInstance(self.wsi.level_dimensions, list)
        self.assertEqual(self.wsi.width, self.wsi.level_dimensions[0][0])
        self.assertEqual(self.wsi.height, self.wsi.level_dimensions[0][1])
        self.assertGreater(self.wsi.mpp, 0)
        self.assertIn("barcode", self.wsi.properties)
        self.assertTrue(hasattr(self.wsi, "mag"))
        self.assertTrue(self.wsi.mag is not None)
        self.assertTrue(callable(self.wsi._fetch_mpp))
        self.assertTrue(callable(self.wsi._fetch_magnification))

    def test_read_region_numpy(self):
        arr = self.wsi.read_region((0, 0), 0, (256, 256), read_as='numpy')
        self.assertEqual(arr.shape, (256, 256, 3))
        self.assertEqual(arr.dtype, self.wsi.img.read_region(0, 0, 256, 256, 0).dtype)

    def test_read_region_pil(self):
        img = self.wsi.read_region((0, 0), 0, (128, 128), read_as='pil')
        from PIL import Image
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (128, 128))

    def test_get_dimensions(self):
        dims = self.wsi.get_dimensions()
        self.assertEqual(dims, self.wsi.img.dimensions)

    def test_get_thumbnail(self):
        thumb = self.wsi.get_thumbnail((64, 64))
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        self.assertEqual(thumb.size, (64, 64))
        self.assertEqual(len(thumb.getbands()), 3)
=======
        self.assertEqual(thumb.shape, (64, 64, 3))
>>>>>>> test basic loading and integration
=======
        self.assertEqual(thumb.size, (64, 64))
        self.assertEqual(len(thumb.getbands()), 3)
>>>>>>> rework test to expect pil image from get_thumbnail test
=======
        self.assertEqual(thumb.shape, (64, 64, 3))
>>>>>>> test basic loading and integration


if __name__ == "__main__":
    unittest.main()
