import traceback
try:
    import test_multi_dist
    print("IMPORTED")
    test_multi_dist.test_marginal_and_heterogeneous()
    print("CALLED AND RETURNED")
except Exception as e:
    print("CAUGHT EXCEPTION")
    traceback.print_exc()
