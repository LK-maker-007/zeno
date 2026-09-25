from __future__ import annotations

import itertools
import unittest

import numpy as np

from scoreboard.entrants import Oracle
from scoreboard.metrics import aurc, auroc, cluster_bootstrap_diff, delong_paired, selective_accuracy
from scoreboard.run import race
from scoreboard.world import N_TRAIN_TEMPLATES, RELATIONS, make_world


def brute_auroc(s, y):
    pos = [a for a, t in zip(s, y, strict=True) if t]
    neg = [a for a, t in zip(s, y, strict=True) if not t]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def brute_delong_var(sa, sb, y):
    # Direct O(mn) structural components (DeLong et al. 1988), to check the fast version.
    def comps(s):
        pos, neg = s[y], s[~y]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        return psi.mean(1), psi.mean(0)

    (a10, a01), (b10, b01) = comps(sa), comps(sb)
    m, n = y.sum(), (~y).sum()
    s10, s01 = np.cov(np.vstack([a10, b10])), np.cov(np.vstack([a01, b01]))
    c = s10 / m + s01 / n
    return c[0, 0] + c[1, 1] - 2 * c[0, 1]


class TestMetrics(unittest.TestCase):
    def test_auroc_matches_pair_counting_with_ties(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            s = rng.integers(0, 5, 60).astype(float)
            y = rng.random(60) < 0.4
            self.assertAlmostEqual(auroc(s, y), brute_auroc(s, y), places=12)

    def test_delong_matches_direct_structural_components(self):
        rng = np.random.default_rng(2)
        y = rng.random(80) < 0.5
        sa = y + rng.normal(0, 1, 80)
        sb = np.round(y + rng.normal(0, 1.5, 80), 1)
        diff, se, _ = delong_paired(sa, sb, y)
        self.assertAlmostEqual(diff, auroc(sa, y) - auroc(sb, y), places=12)
        self.assertAlmostEqual(se**2, brute_delong_var(sa, sb, y), places=12)

    def test_selective_accuracy_and_aurc_by_hand(self):
        s = [0.9, 0.8, 0.7, 0.1]
        y = [True, False, True, False]
        self.assertAlmostEqual(selective_accuracy(s, y, 0.5), 0.5)
        self.assertAlmostEqual(selective_accuracy(s, y, 0.75), 2 / 3)
        self.assertAlmostEqual(aurc(s, y), np.mean([0 / 1, 1 / 2, 1 / 3, 2 / 4]))

    def test_constant_confidence_earns_only_the_base_rate(self):
        y = [True] * 30 + [False] * 70
        s = [0.5] * 100
        self.assertAlmostEqual(selective_accuracy(s, y, 0.5), 0.3)
        self.assertAlmostEqual(aurc(s, y), 0.7)
        self.assertAlmostEqual(aurc(s, y), aurc(s, y[::-1]))

    def test_cluster_bootstrap_returns_nan_bounds_when_no_resample_is_defined(self):
        d, lo, hi = cluster_bootstrap_diff(auroc, ([0.1, 0.2], [True, True]), ([0.3, 0.4], [True, True]), [0, 1])
        self.assertTrue(np.isnan(d) and np.isnan(lo) and np.isnan(hi))

    def test_cluster_bootstrap_zero_for_identical_inputs(self):
        rng = np.random.default_rng(3)
        s, y = rng.random(50), rng.random(50) < 0.5
        d, lo, hi = cluster_bootstrap_diff(auroc, (s, y), (s, y), rng.integers(0, 10, 50), n_boot=200)
        self.assertEqual((d, lo, hi), (0.0, 0.0, 0.0))


class TestWorld(unittest.TestCase):
    def setUp(self):
        self.w = make_world(0)

    def test_same_seed_same_digest_other_seed_differs(self):
        self.assertEqual(self.w.digest(), make_world(0).digest())
        self.assertNotEqual(self.w.digest(), make_world(1).digest())

    def test_gold_labels_consistent_with_facts(self):
        facts = {(f.subject, f.relation): f.value for f in self.w.facts}
        people = {f.subject for f in self.w.facts}
        for p in self.w.probes:
            key = (p.subject, p.relation)
            if p.stratum in ("i_known", "iv_paraphrase"):
                self.assertEqual(p.gold, facts[key])
            else:
                self.assertIsNone(p.gold)
                self.assertNotIn(key, facts)
            if p.stratum == "iii_unseen_subject":
                self.assertNotIn(p.subject, people)

    def test_near_miss_has_a_same_family_fact(self):
        by_subject = {}
        for f in self.w.facts:
            by_subject.setdefault(f.subject, set()).add(RELATIONS[f.relation][0])
        near = [p for p in self.w.probes if p.stratum == "ii_near_miss"]
        self.assertTrue(near)
        for p in near:
            self.assertIn(RELATIONS[p.relation][0], by_subject[p.subject])

    def test_heldout_templates_never_used_for_training(self):
        train_questions = {
            q.format(s=p.subject)
            for p in self.w.probes
            if p.stratum == "iv_paraphrase"
            for q in RELATIONS[p.relation][3][:N_TRAIN_TEMPLATES]
        }
        for p in self.w.probes:
            if p.stratum == "iv_paraphrase":
                self.assertNotIn(p.question, train_questions)
        heldout = set(self.w.heldout_statements)
        for f in self.w.facts:
            self.assertNotIn(f.text, heldout)

    def test_every_stratum_populated(self):
        strata = {p.stratum for p in self.w.probes}
        self.assertEqual(strata, {"i_known", "ii_near_miss", "ii_far_miss", "iii_unseen_subject", "iv_paraphrase"})


class TestRace(unittest.TestCase):
    def test_oracle_is_perfect(self):
        w = make_world(0)
        r = race(Oracle(w.facts, w.probes), w, verbose=False)
        self.assertEqual(r["T1_recall_known"], 1.0)
        self.assertEqual(r["T1_recall_paraphrase"], 1.0)
        self.assertEqual(r["T3_auroc"], 1.0)
        self.assertEqual(r["T2_forgetting"], 0.0)


if __name__ == "__main__":
    unittest.main()
