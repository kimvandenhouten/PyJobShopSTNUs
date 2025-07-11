************************************************************************
file with basedata            : mm4_.bas
initial value random generator: 1441018644
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
    1     10      0       12        7       12
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           6
   3        3          3           5   8  10
   4        3          3           6   7  11
   5        3          1           6
   6        3          1           9
   7        3          2           8  10
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
  2      1     5       2    0    0    5
         2     7       0    6    5    0
         3     8       0    5    4    0
  3      1     2       7    0    8    0
         2     3       7    0    0    1
         3     9       3    0    0    1
  4      1     4       0    8    6    0
         2     4       4    0    8    0
         3     9       3    0    6    0
  5      1     5       7    0    0    7
         2     8       0    5    0    4
         3     9       0    2    0    3
  6      1     2       8    0    4    0
         2     6       7    0    4    0
         3    10       6    0    2    0
  7      1     2       0    4    0    7
         2     4       3    0    0    4
         3     6       0    3    0    2
  8      1     1       8    0    0    7
         2     2       0    4    0    4
         3     7       4    0    2    0
  9      1     3       6    0    0    2
         2     5       5    0    2    0
         3     8       4    0    2    0
 10      1     2       0    5    0    5
         2     9       2    0    2    0
         3    10       0    2    0    2
 11      1     2       0    9    0    4
         2     5       0    7    0    4
         3     8       4    0    2    0
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    9    5   33   38
************************************************************************
DEADLINES:
jobnr.  deadline
  2       49
  3       56
  4       36
  6       67
  7       58
  8       65
  9       65
  10      30
  11      57
************************************************************************
