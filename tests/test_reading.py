from __future__ import annotations

import collections
import unittest

from scoreboard.reading import CATEGORIES, MAX_FACTS, UNKNOWN, build, digest
from scoreboard.regex_reader import RegexReader


class TestReading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build(range(900, 905), 200, CATEGORIES, seed=1)

    def test_labels_match_a_reader_that_knows_every_template(self):
        oracle = RegexReader(all_templates=True)
        wrong = collections.Counter()
        for e in self.data:
            if oracle.read(e.facts, e.question)[0] != e.answer:
                wrong[e.category] += 1
        self.assertEqual(dict(wrong), {})

    def test_deterministic(self):
        self.assertEqual(digest(self.data), digest(build(range(900, 905), 200, CATEGORIES, seed=1)))
        self.assertNotEqual(digest(self.data), digest(build(range(900, 905), 200, CATEGORIES, seed=2)))

    def test_context_bounds_and_both_classes_per_category(self):
        by_cat = collections.defaultdict(collections.Counter)
        for e in self.data:
            self.assertLessEqual(len(e.facts), MAX_FACTS)
            by_cat[e.category][e.answer == UNKNOWN] += 1
        for c in ("simple", "same_name", "two_hop", "heldout_question", "heldout_fact", "heldout_both"):
            self.assertGreater(by_cat[c][True], 0, c)
            self.assertGreater(by_cat[c][False], 0, c)
        self.assertEqual(by_cat["negation"][False], 0)
        self.assertEqual(by_cat["near_miss"][False], 0)
        self.assertEqual(by_cat["update"][True], 0)

    def test_training_categories_use_only_training_wording(self):
        floor = RegexReader(all_templates=False)
        train_only = build(range(1000, 1003), 100, [c for c in CATEGORIES if not c.startswith("heldout")], seed=3)
        for e in train_only:
            self.assertEqual(len(floor._records(e.facts)), len(e.facts), e.facts)
            self.assertTrue(any(pat.match(e.question) for _, pat in floor.questions), e.question)

    def test_auroc_counts_only_correct_real_answers_as_positive(self):
        from scoreboard.read_race import score

        # Knows every wording: right everywhere and confident only in real answers, so AUROC must be exactly 1.
        m = score(RegexReader(all_templates=True), self.data)["categories"]["ALL"]
        self.assertEqual((m["acc"], m["auroc"]), (1.0, 1.0))

        class Abstainer:
            name = "always-unknown"

            def read(self, facts, question):
                return UNKNOWN, 0.0

        # Right on every unanswerable question, yet it holds no real answer, so AUROC has no positives.
        m = score(Abstainer(), self.data)["categories"]["ALL"]
        self.assertAlmostEqual(m["acc"], sum(e.answer == UNKNOWN for e in self.data) / len(self.data))
        self.assertTrue(m["auroc"] != m["auroc"])

    def test_heldout_categories_use_heldout_wording(self):
        floor = RegexReader(all_templates=False)
        for e in self.data:
            if e.category in ("heldout_question", "heldout_both"):
                self.assertFalse(any(pat.match(e.question) for _, pat in floor.questions), e.question)
            if e.category in ("heldout_fact", "heldout_both"):
                self.assertEqual(floor._records(e.facts), [], e.facts)


if __name__ == "__main__":
    unittest.main()
