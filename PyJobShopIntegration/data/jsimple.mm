************************************************************************
file with basedata            : simple_example.bas
initial value random generator: 123456789
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  6
horizon                       : 20
RESOURCES
  - renewable                 :  1   R
  - nonrenewable              :  1   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1      4     0       10        2       10
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          2           2   3
   2        2          1           4
   3        2          1           5
   4        2          1           6
   5        2          1           6
   6        1          0
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  N 1
------------------------------------------------------------------------
  1      1     0       0    0
  2      1     3       2    4
         2     5       1    2
  3      1     2       3    2
         2     4       1    3
  4      1     2       2    3
         2     3       1    2
  5      1     1       2    3
         2     2       1    2
  6      1     0       0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  N 1
    4   10
************************************************************************
