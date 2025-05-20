************************************************************************
file with basedata            : dc_safe_example.bas
initial value random generator: 123456789
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  5
horizon                       : 20
RESOURCES
  - renewable                 :  1   R
  - nonrenewable              :  0
  - doubly constrained        :  0
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1      3     0       15        0       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          1           2
   2        1          2           3   4
   3        1          1           5
   4        1          1           5
   5        1          0
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1
------------------------------------------------------------------------
  1      1     0       0
  2      1     3       2
  3      1     2       1
  4      1     3       1
  5      1     0       0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1
    3
************************************************************************
