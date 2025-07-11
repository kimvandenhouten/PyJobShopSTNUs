************************************************************************
file with basedata            : mm55_.bas
initial value random generator: 1396876967
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
    1     10      0       24        7       24
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           5
   3        3          3           5   6  10
   4        3          3           6   7  10
   5        3          1           8
   6        3          1          11
   7        3          2           8   9
   8        3          1          11
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       9    5    8    2
         2     6       8    3    7    2
         3     8       6    3    6    2
  3      1     4       6    8    7    8
         2     5       5    6    6    8
         3     9       3    4    6    7
  4      1     1       5    3    7    9
         2     1       8    2    7    9
         3     5       3    2    6    9
  5      1     1       3    4    5    8
         2     5       3    3    5    6
         3     8       2    2    4    2
  6      1     6       8    6    6    6
         2     7       7    6    4    5
         3     8       7    6    2    5
  7      1     8      10    7    9    4
         2     9       8    7    8    4
         3     9       7    7    9    2
  8      1     8       4    4    4    5
         2    10       4    4    1    3
         3    10       3    4    1    5
  9      1     3       7    8    9    4
         2     5       4    8    8    2
         3    10       2    8    8    1
 10      1     3       6    3    9    9
         2     3       7    3    8   10
         3     8       6    1    5    8
 11      1     7       8    7    6    8
         2     8       5    7    4    7
         3     9       5    6    3    7
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   22   16   64   59
************************************************************************
DEADLINES:
jobnr.  deadline
  3       38
  6       53
  11      26
************************************************************************
