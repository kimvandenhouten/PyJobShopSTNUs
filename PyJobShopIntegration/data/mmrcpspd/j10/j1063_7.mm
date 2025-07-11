************************************************************************
file with basedata            : mm63_.bas
initial value random generator: 943036505
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  80
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       15        2       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           7
   3        3          1           8
   4        3          3           5   6   7
   5        3          3           9  10  11
   6        3          1           8
   7        3          1           8
   8        3          2           9  10
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     5       5    7    6    6
         2     9       5    7    5    6
         3    10       4    7    5    5
  3      1     2       9    6    7    5
         2     3       9    5    6    4
         3     8       8    3    3    3
  4      1     1       2    9    5    7
         2     5       2    8    4    5
         3     7       2    3    2    1
  5      1     1       9    5    9    4
         2     3       8    5    6    3
         3     4       7    3    4    2
  6      1     1       9    3    3    5
         2     2       7    2    2    4
         3     8       3    2    1    3
  7      1     6       6    2    8    9
         2     6       4    2    9    9
         3     8       2    2    8    9
  8      1     2       7    8   10    6
         2     5       7    7    8    4
         3     9       4    4    7    4
  9      1     1       9    4    9    2
         2     5       8    4    9    1
         3    10       8    4    7    1
 10      1     2       9    7    7    7
         2     3       5    6    6    5
         3     9       4    5    5    4
 11      1     1       6    5    6    6
         2     7       4    3    5    5
         3     7       4    4    5    4
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   32   22   64   51
************************************************************************
DEADLINES:
jobnr.  deadline
  2       14
  3       51
  6       1
  10      10
************************************************************************
