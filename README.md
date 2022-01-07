# B-n-B algorithms for PCGTSP

Here come source code, input data and results
of numerical experiments for solving
Precedence Constrained General Traveling Salesman Problem
(PCGTSP)
referred to in
[Problem-Specific Branch-and-Bound Algorithms for the Precedence Constrained Generalized Traveling Salesman Problem][optima2021].

## Prerequisites

One need modern installation of [Python]
as well as modules in
[requirements.txt](requirements.txt):
```sh
pip install -r requirements.txt
```

For DP version to run,
working installation of [Gurobi]
along with appropriate license
(the academic one is just fine)
is recommended.

## Test data

Problem instances are in
[Salman/input](Salman/input)
in PCGLNS format.

Corresponding solutions
by PCGLNS heuristic
are in
[Salman/heuristic](Salman/heuristic).

Logs of B-n-B flavoure of the algorithm
are in
[logs/s](logs/s),
of DP flavour -
in
[logs/salman](logs/salman).

[optima2021]: https://link.springer.com/chapter/10.1007/978-3-030-91059-4_10
[Python]: https://www.python.org/
[Gurobi]: https://www.gurobi.com/
