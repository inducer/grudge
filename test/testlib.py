# Needed here because MPI test orchestration imports the test module twice,
# leading to two nominally different tag types. Grrr.

class SimpleTag:
    pass
