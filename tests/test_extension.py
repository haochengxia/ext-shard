import xtensor_shard as m
from unittest import TestCase


class ExampleTest(TestCase):
    nl = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    ss = m.ShardedStructure(nl)

    def test_basic(self):
        self.assertEqual(4, self.ss.get_max_shard_size())
        self.assertEqual(10, self.ss.get_num_ele())
        self.assertEqual(3, self.ss.get_num_shard())

    def test_sample_flat(self):
        perm = self.ss.sample_perm_flat()
        self.assertEqual(10, len(perm))

    def test_sample_nest(self):
        perm = self.ss.sample_perm_nest()
        self.assertEqual(3, len(perm))

    def test_wr_idxes_available(self):
        idxes = self.ss.idxes_available_
        self.assertEqual(10, len(idxes))
        self.ss.idxes_available_ = [0, 1, 2]
        idxes = self.ss.idxes_available_
        self.assertEqual(3, len(idxes))
