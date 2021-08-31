# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import unittest

from summa import highlighting


class TestMatchMostText(unittest.TestCase):
    def test_happy_path(self):
        summary = """ Knitting creates stitches: loops of yarn in a row, either flat or in the round (tubular) There are usually many active stitches on the knitting needle at one time . Knitted fabric consists of consecutive rows of connected loops that intermesh with the next and previous rows ."""
        article = """Knitting is a method by which yarn is manipulated to create a textile or fabric. It is used in many types of garments. Knitting may be done by hand or by machine.
Knitting creates stitches: loops of yarn in a row, either flat or in the round (tubular). There are usually many active stitches on the knitting needle at one time. Knitted fabric consists of a number of consecutive rows of connected loops that intermesh with the next and previous rows.  As each row is formed, each newly created loop is pulled through one or more loops from the prior row and placed on the gaining needle so that the loops from the prior row can be pulled off the other needle without unraveling.
Differences in yarn (varying in fibre type, weight, uniformity and twist), needle size, and stitch type allow for a variety of knitted fabrics with different properties, including color, texture, thickness, heat retention, water resistance, and integrity. A small sample of knitwork is known as a swatch."""

        actual = highlighting.match_most_text(summary, article)
        expected = ["Knitting creates stitches: loops of yarn in a row, either flat or in the round (tubular). There are usually many active stitches on the knitting needle at one time. Knitted fabric consists of", 
                    "consecutive rows of connected loops that intermesh with the next and previous rows.",]

        self.assertEqual(expected, actual)

    def test_no_punctuation(self):
        """thing"""
        summary = "quick brown lazy dog"
        article = "the quick brown fox jumped over the lazy dog"
        actual = highlighting.match_most_text(summary, article)
        expected = ["quick brown", "lazy dog"]

        self.assertEqual(expected, actual)
