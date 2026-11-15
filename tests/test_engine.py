"""Independent dominance oracle and physical roster checks."""
import unittest
import numpy as np
from src.engine import pareto_mask, brute_pareto

class ParetoTests(unittest.TestCase):
    def test_ties_and_duplicates(self):
        v=np.array([[1,2],[1,2],[1,1],[2,2],[3,4]])
        self.assertEqual(pareto_mask(v).tolist(),[True,True,False,False,True])
    def test_all_directions_against_oracle(self):
        rng=np.random.default_rng(51)
        for directions in [(False,True),(True,False),(True,True),(False,False)]:
            for _ in range(30):
                v=rng.integers(-10,11,(70,2))
                np.testing.assert_array_equal(pareto_mask(v,directions),brute_pareto(v,directions))
    def test_input_edges(self):
        self.assertEqual(pareto_mask(np.empty((0,2))).size,0)
        self.assertTrue(pareto_mask([[1,1]])[0])
        for invalid in [[[np.nan,1]],[[1,np.inf]],[[1,2,3]]]:
            with self.assertRaises(ValueError):pareto_mask(invalid)

if __name__=='__main__':unittest.main()
