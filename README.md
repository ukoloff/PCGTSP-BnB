# B-n-B algorithms for PCGTSP

Here come source code, input data and results
of numerical experiments for solving
Precedence Constrained General Traveling Salesman Problem
(PCGTSP)
referred to in
[Problem-Specific Branch-and-Bound Algorithms for the Precedence Constrained Generalized Traveling Salesman Problem][optima2021].

[optima2021]: https://link.springer.com/chapter/10.1007/978-3-030-91059-4_10

## Prerequisites

One needs modern installation of [Python]
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

[Python]: https://www.python.org/
[Gurobi]: https://www.gurobi.com/

## Test data

Problem instances are in
[Salman/input](Salman/input)
in PCGLNS format.

Corresponding solutions
by PCGLNS heuristic
are in
[Salman/heuristic](Salman/heuristic).

Logs of B-n-B flavour of the algorithm
are in
[logs/s](logs/s),
of DP flavour -
in
[logs/salman](logs/salman).

## Running algorithms

### Branch-and-Bound flavour

Source code for B-n-B design is in
[playground/](playground).
To run:
```sh
python path/to/walk.py instance
```
Use instance name (eg `e3x_1`)
without path or extension.
Result of PCGLNS evaluation will be used
automagically.

Logs will fall into
[logs/](logs).

### Dynamic Programming flavour

Source code for DP design is in
[dp/](dp).
To run:
```sh
python path/to/DP_pcglns.py -i=path/to/instance.pcglns -UB=NNNN -w=1
```
Specify path to PCGLNS file (with extension),
best solution by PCGLNS
and optionally number of parallel workers to run.

Logs will go to standard output / error.
