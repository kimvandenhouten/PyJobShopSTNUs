************************************************************************
file with basedata            : mm3_.bas
initial value random generator: 632771121
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  83
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       10        7       10
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1          11
   3        3          2           5   6
   4        3          1           8
   5        3          2           9  10
   6        3          3           7   8  11
   7        3          2           9  10
   8        3          1           9
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       0    6    0    7
         2     6       2    0    0    6
         3    10       0    5    5    0
  3      1     1       0    4    4    0
         2     2       0    4    0    6
         3     6       0    3    2    0
  4      1     2       0    6    0    9
         2     2       7    0    0    9
         3    10       2    0    0    8
  5      1     1       5    0    0    6
         2     1       6    0    3    0
         3     6       4    0    0    8
  6      1     3       0    5    0    4
         2     8       7    0    0    3
         3     9       7    0    1    0
  7      1     4       0    9    0    3
         2     5       0    6    5    0
         3     9       8    0    0    1
  8      1     1       0    6    8    0
         2     6       0    2    7    0
         3    10       9    0    6    0
  9      1     2      10    0    0    8
         2     3       8    0    3    0
         3     8       7    0    0    7
 10      1     1       0    6    7    0
         2     3       0    1    0    5
         3     6       7    0    5    0
 11      1     3       0    4    8    0
         2     7       0    2    0    6
         3     9       6    0    8    0
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   10    6   33   42
************************************************************************
DEADLINES:
jobnr.  deadline
  2       25
  3       72
  4       14
  5       82
  6       22
  8       47
  9       43
  10      54
  11      42
************************************************************************
