import pytest

from src.symbolic import (
    Decomposability,
    Determinism,
    Distribution,
    GaussianDistribution,
    ProductNode,
    Smoothness,
    StructuredDecomposability,
    SumNode,
    SymbolicArithmeticCircuit,
)
from src.utils import BitSet, ContinuousInterval, Support


def support_from_scope(scope: BitSet) -> Support:
    intervals = {i: ContinuousInterval(float("-inf"), float("inf")) for i in scope}
    return Support(intervals)


class TestProperties:
    @pytest.fixture
    def circuit(self):
        return SymbolicArithmeticCircuit()

    def test_smoothness_trivial(self, circuit):
        # Leaf node is smooth
        leaf = GaussianDistribution(0, 0, 1)
        circuit.add_node(0, leaf)
        assert Smoothness().check(0, circuit)

        # Product node is smooth (no condition)
        circuit.add_node(1, ProductNode(support_from_scope(BitSet([0, 1]))))
        assert Smoothness().check(1, circuit)

    def test_smoothness_sum_node(self, circuit):
        # Valid: children have same scope
        child1 = GaussianDistribution(0, 0, 1)
        child2 = GaussianDistribution(0, 1, 1)

        circuit.add_node(0, child1)
        circuit.add_node(1, child2)

        sum_node = SumNode(support_from_scope(BitSet([0])))
        circuit.add_node(2, sum_node)
        circuit.add_edge(2, 0)
        circuit.add_edge(2, 1)

        assert Smoothness().check(2, circuit)

    def test_smoothness_invalid(self, circuit):
        # Invalid: children have different scopes
        child1 = GaussianDistribution(0, 0, 1)
        child2 = GaussianDistribution(1, 0, 1)

        circuit.add_node(0, child1)
        circuit.add_node(1, child2)

        sum_node = SumNode(support_from_scope(BitSet([0, 1])))
        circuit.add_node(2, sum_node)
        circuit.add_edge(2, 0)
        circuit.add_edge(2, 1)

        assert not Smoothness().check(2, circuit)

    def test_decomposability_trivial(self, circuit):
        # Leaf is decomposable
        leaf = GaussianDistribution(0, 0, 1)
        circuit.add_node(0, leaf)
        assert Decomposability().check(0, circuit)

        # Sum node is decomposable
        circuit.add_node(1, SumNode(support_from_scope(BitSet([0]))))
        assert Decomposability().check(1, circuit)

    def test_decomposability_product_node(self, circuit):
        # Valid: disjoint scopes
        child1 = GaussianDistribution(0, 0, 1)
        child2 = GaussianDistribution(1, 0, 1)

        circuit.add_node(0, child1)
        circuit.add_node(1, child2)

        prod_node = ProductNode(support_from_scope(BitSet([0, 1])))
        circuit.add_node(2, prod_node)
        circuit.add_edge(2, 0)
        circuit.add_edge(2, 1)

        assert Decomposability().check(2, circuit)

    def test_decomposability_invalid(self, circuit):
        # Invalid: overlapping scopes
        child1 = GaussianDistribution(0, 0, 1)  # scope {0}
        child2 = GaussianDistribution(1, 0, 1)  # scope {1}

        # We need overlapping scopes.
        # Create children that are ProductNodes of (0,1) and (1,2)

        # Child 1: Product over 0 and 1
        c1_0 = GaussianDistribution(0, 0, 1)
        c1_1 = GaussianDistribution(1, 0, 1)
        circuit.add_node(10, c1_0)
        circuit.add_node(11, c1_1)
        node_01 = ProductNode(support_from_scope(BitSet([0, 1])))
        circuit.add_node(0, node_01)
        circuit.add_edge(0, 10)
        circuit.add_edge(0, 11)

        # Child 2: Product over 1 and 2
        c2_1 = GaussianDistribution(1, 0, 1)
        c2_2 = GaussianDistribution(2, 0, 1)
        circuit.add_node(21, c2_1)
        circuit.add_node(22, c2_2)
        node_12 = ProductNode(support_from_scope(BitSet([1, 2])))
        circuit.add_node(1, node_12)
        circuit.add_edge(1, 21)
        circuit.add_edge(1, 22)

        prod_node = ProductNode(support_from_scope(BitSet([0, 1, 2])))
        circuit.add_node(2, prod_node)
        circuit.add_edge(2, 0)
        circuit.add_edge(2, 1)

        assert not Decomposability().check(2, circuit)

    def test_determinism_trivial(self, circuit):
        # Leaf is deterministic
        leaf = GaussianDistribution(0, 0, 1)
        circuit.add_node(0, leaf)
        assert Determinism().check(0, circuit)

        # Product is deterministic
        circuit.add_node(1, ProductNode(support_from_scope(BitSet([0]))))
        assert Determinism().check(1, circuit)

    def test_determinism_sum_node_valid(self, circuit):
        class MockDistribution(Distribution):
            def sample(self):
                pass

            def split_at(self, cut_point):
                pass

            def constrain_to(self, interval):
                pass

        # Use var index and interval directly
        n1 = MockDistribution(0, ContinuousInterval(0, 1))
        n2 = MockDistribution(0, ContinuousInterval(1, 2))

        circuit.add_node(0, n1)
        circuit.add_node(1, n2)
        sum_node = SumNode(support_from_scope(BitSet([0])))

        circuit.add_node(2, sum_node)
        circuit.add_edge(2, 0)
        circuit.add_edge(2, 1)

        assert Determinism().check(2, circuit)

    def test_structured_decomposability(self, circuit):
        sd = StructuredDecomposability()

        circuit.add_node(0, GaussianDistribution(0, 0, 1))
        circuit.add_node(1, GaussianDistribution(1, 0, 1))

        circuit.add_node(2, ProductNode(support_from_scope(BitSet([0, 1]))))
        circuit.add_edge(2, 0)  # {0}
        circuit.add_edge(2, 1)  # {1}

        circuit.add_node(3, ProductNode(support_from_scope(BitSet([0, 1]))))
        circuit.add_edge(3, 0)
        circuit.add_edge(3, 1)

        assert sd.check(2, circuit)
        assert sd.check(3, circuit)

    def test_structured_decomposability_invalid(self, circuit):
        sd = StructuredDecomposability()
        # Case 2: Invalid structure
        # P1({0,1,2}) -> {0}, {1,2}
        # P2({0,1,2}) -> {0,1}, {2}

        l0 = GaussianDistribution(0, 0, 1)
        l1 = GaussianDistribution(1, 0, 1)
        l2 = GaussianDistribution(2, 0, 1)

        circuit.add_node(0, l0)
        circuit.add_node(1, l1)
        circuit.add_node(2, l2)

        # Helper for "intermediate" products
        p12 = ProductNode(support_from_scope(BitSet([1, 2])))
        circuit.add_node(12, p12)
        circuit.add_edge(12, 1)
        circuit.add_edge(12, 2)

        p01 = ProductNode(support_from_scope(BitSet([0, 1])))
        circuit.add_node(101, p01)
        circuit.add_edge(101, 0)
        circuit.add_edge(101, 1)

        # P1 root
        P1 = ProductNode(support_from_scope(BitSet([0, 1, 2])))
        circuit.add_node(100, P1)
        circuit.add_edge(100, 0)  # {0}
        circuit.add_edge(100, 12)  # {1,2}

        # P2 root
        P2 = ProductNode(support_from_scope(BitSet([0, 1, 2])))
        circuit.add_node(200, P2)
        circuit.add_edge(200, 101)  # {0,1}
        circuit.add_edge(200, 2)  # {2}

        assert not sd.check(100, circuit)
