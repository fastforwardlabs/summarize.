import unittest
import utils 


class TestMatchMostText(unittest.TestCase):
    def test_happy_path(self):
        """ thing """
        summary = """ Knitting creates stitches: loops of yarn in a row, either flat or in the round (tubular) There are usually many active stitches on the knitting needle at one time . Knitted fabric consists of consecutive rows of connected loops that intermesh with the next and previous rows ."""
        article = """Knitting is a method by which yarn is manipulated to create a textile or fabric. It is used in many types of garments. Knitting may be done by hand or by machine.
Knitting creates stitches: loops of yarn in a row, either flat or in the round (tubular). There are usually many active stitches on the knitting needle at one time. Knitted fabric consists of a number of consecutive rows of connected loops that intermesh with the next and previous rows.  As each row is formed, each newly created loop is pulled through one or more loops from the prior row and placed on the gaining needle so that the loops from the prior row can be pulled off the other needle without unraveling.
Differences in yarn (varying in fibre type, weight, uniformity and twist), needle size, and stitch type allow for a variety of knitted fabrics with different properties, including color, texture, thickness, heat retention, water resistance, and integrity. A small sample of knitwork is known as a swatch."""
        
        actual = utils.match_most_text(summary, article)
        expected = ["Knitting creates stitches: loops of yarn in a row, either flat or in the round (tubular). There are usually many active stitches on the knitting needle at one time. Knitted fabric consists of", 
                    "consecutive rows of connected loops that intermesh with the next and previous rows.",]

        self.assertEqual(expected, actual)

    def test_no_punctuation(self):
        """ thing """
        summary = "quick brown lazy dog"
        article = "the quick brown fox jumped over the lazy dog"
        actual = utils.match_most_text(summary, article)
        expected = ["quick brown", 
                    "lazy dog"]

        self.assertEqual(expected, actual)