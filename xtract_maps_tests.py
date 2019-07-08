from xtract_maps_main import extract_map_metadata
import os
import unittest

current_directory = os.getcwd()

img1 = current_directory + '/test_imgs/CAIBOX_2009_map.jpg'
img2 = current_directory + '/test_imgs/Bigelow2015_map.jpg'
img3 = current_directory + '/test_imgs/GOMECC2_map.jpg'
img4 = current_directory + '/test_imgs/Marion_Dufresne_map_1991_1993.jpg'
img5 = current_directory + '/test_imgs/Oscar_Dyson_map.jpg'
img6 = current_directory + '/test_imgs/P16S_2014_map.jpg'
img7 = current_directory + '/test_imgs/us_states.png'


# Test cases for xtract-maps. Very naively only checks whether it outputs
# something or not.
class MapTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_img1(self):
        img1_metadata = extract_map_metadata(img1)
        for i in range(3):
            self.assertTrue(img1_metadata[i])

    def test_img2(self):
        img2_metadata = extract_map_metadata(img2)
        self.assertTrue(img2_metadata[0])
        self.assertTrue(img2_metadata[1])
        self.assertFalse(img2_metadata[2])

    def test_img3(self):
        img3_metadata = extract_map_metadata(img3)
        self.assertTrue(img3_metadata[0])
        self.assertTrue(img3_metadata[1])
        self.assertFalse(img3_metadata[2])

    def test_img4(self):
        img4_metadata = extract_map_metadata(img4)
        self.assertTrue(img4_metadata[0])
        self.assertTrue(img4_metadata[1])
        self.assertFalse(img4_metadata[2])

    def test_img5(self):
        img5_metadata = extract_map_metadata(img5)
        self.assertTrue(img5_metadata[0])
        self.assertTrue(img5_metadata[1])
        self.assertFalse(img5_metadata[2])

    def test_img6(self):
        img6_metadata = extract_map_metadata(img6)
        for i in range(3):
            self.assertTrue((img6_metadata[i]))

    def test_img7(self):
        img7_metadata = extract_map_metadata(img7)
        self.assertFalse(img7_metadata[0])
        self.assertFalse(img7_metadata[1])
        self.assertTrue(img7_metadata[2])


if __name__ == "__main__":
    unittest.main()
