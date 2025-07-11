************************************************************************
file with basedata            : mm7_.bas
initial value random generator: 1892276334
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
    1     10      0       19        0       19
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5  10
   3        3          2           5   7
   4        3          2           9  10
   5        3          1           6
   6        3          1           9
   7        3          1           8
   8        3          3           9  10  11
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     6       9    0    6    9
         2     6       0    4    5    9
         3     8       9    0    3    6
  3      1     4       0    7    7    5
         2     6       0    6    7    4
         3     9       0    5    4    3
  4      1     1       5    0    8    6
         2     6       5    0    7    5
         3    10       3    0    5    5
  5      1     2       0    8    5    9
         2     3       0    7    5    6
         3     5       7    0    4    3
  6      1     3       0    5    8    5
         2     6       0    4    6    3
         3     6       5    0    5    1
  7      1     5       8    0    6    8
         2     8       0    9    3    8
         3    10       4    0    3    6
  8      1     4       2    0    5    7
         2     7       0    6    4    6
         3    10       0    4    4    5
  9      1     2       5    0   10    6
         2     4       2    0    8    3
         3    10       0    7    8    3
 10      1     6       4    0    1    9
         2     6       0    6    1    6
         3     8       0    6    1    3
 11      1     2       0    6    7    7
         2     2       1    0    6    7
         3     7       0    5    6    7
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    6    8   57   62
************************************************************************
DEADLINES:
jobnr.  deadline
  2       42
  3       62
  4       38
  5       30
  6       68
  7       74
  8       18
  9       83
  10      71
  11      41
************************************************************************
