************************************************************************
file with basedata            : mm5_.bas
initial value random generator: 652647244
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  84
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       17        0       17
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   6
   3        3          1           7
   4        3          1           8
   5        3          2           7  11
   6        3          3           8  10  11
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
  2      1     2       7    0    8    5
         2     6       7    0    5    4
         3     6       6    0    5    5
  3      1     7       0    5    2    7
         2     8       0    4    2    7
         3    10       0    4    2    6
  4      1     1       9    0    7   10
         2     4       9    0    5    9
         3     9       9    0    4    9
  5      1     1       5    0    7   10
         2     7       0    5    5   10
         3    10       0    4    4   10
  6      1     1       4    0    5    5
         2     1       0    8    5    7
         3     5       0    5    4    4
  7      1     6       0    4    5    8
         2     8       0    4    3    6
         3    10       0    4    3    5
  8      1     1       0    7    8    7
         2    10       7    0    7    5
         3    10       5    0    7    6
  9      1     1       0    7    5    8
         2     8       7    0    5    7
         3    10       0    2    2    5
 10      1     4       8    0    9    6
         2     6       8    0    4    5
         3     9       0    7    1    3
 11      1     2       0    9    8    7
         2     3       7    0    8    6
         3     5       7    0    7    3
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   10    7   44   58
************************************************************************
DEADLINES:
jobnr.  deadline
  2       56
  4       72
  6       40
  7       43
  8       36
  9       35
  10      17
  11      79
************************************************************************
