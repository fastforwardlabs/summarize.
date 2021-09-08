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

from summa import text_cleanup

ALICE_TEXT = """
“Perhaps it doesn’t understand English,” thought Alice; “I daresay it’s
a French mouse, come over with William the Conqueror.” (For, with all
her knowledge of history, Alice had no very clear notion how long ago
anything had happened.) So she began again: “Où est ma chatte?” which
was the first sentence in her French lesson-book. The Mouse gave a
sudden leap out of the water, and seemed to quiver all over with
fright. “Oh, I beg your pardon!” cried Alice hastily, afraid that she
had hurt the poor animal’s feelings. “I quite forgot you didn’t like
cats.”
""".replace("\n", " ")


class TestMatchMostText(unittest.TestCase):
    def test_unchanged_normal_text(self):
        raw_text = ALICE_TEXT
        expected_text = ALICE_TEXT
        self.assertEqual(text_cleanup.cleanup(raw_text), expected_text)

    def test_fix_missing_space_period(self):
        raw_text = ALICE_TEXT.replace(
            "French lesson-book. The Mouse", "French lesson-book.The Mouse"
        )
        expected_text = ALICE_TEXT
        self.assertEqual(text_cleanup.cleanup(raw_text), expected_text)

    def test_fix_missing_space_exclamation(self):
        exclaimative = ALICE_TEXT.replace(
            "anything had happened.)", "anything had happened!"
        )
        raw_text = exclaimative.replace(
            "anything had happened! So she", "anything had happened!So she"
        )
        expected_text = exclaimative
        self.assertEqual(text_cleanup.cleanup(raw_text), expected_text)

    def test_fix_extra_space_period(self):
        raw_text = ALICE_TEXT.replace(
            "French lesson-book. The Mouse", "French lesson-book . The Mouse"
        )
        expected_text = ALICE_TEXT
        self.assertEqual(text_cleanup.cleanup(raw_text), expected_text)

