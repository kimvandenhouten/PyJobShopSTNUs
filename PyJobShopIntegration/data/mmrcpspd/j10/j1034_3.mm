************************************************************************
file with basedata            : mm34_.bas
initial value random generator: 632364093
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  74
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       11        2       11
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           7   9
   3        3          2           8  11
   4        3          2           5   6
   5        3          2           8  10
   6        3          1          10
   7        3          2          10  11
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
  2      1     4       2    5    6    0
         2     6       2    4    0    6
         3    10       1    2    0    3
  3      1     1       6    7    0    8
         2     4       6    7    4    0
         3     5       5    4    0    5
  4      1     4      10   10    0    9
         2     6       6    9    3    0
         3     6       8    7    0    8
  5      1     4       5    7    0    5
         2     6       5    7    8    0
         3    10       4    7    0    2
  6      1     1       9    8    0    9
         2     3       6    7    5    0
         3     6       4    6    0    7
  7      1     3       9    8    0    9
         2     5       7    8    0    4
         3    10       6    8    4    0
  8      1     1       6    8    0    4
         2     2       5    8    0    3
         3     3       3    5    0    3
  9      1     2       7    3    2    0
         2     7       3    2    0    1
         3     9       3    1    0    1
 10      1     2       7    9    9    0
         2     6       5    7    9    0
         3     9       4    3    8    0
 11      1     3       8    8    9    0
         2     5       6    8    8    0
         3     6       5    7    0    8
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    9   11   29   31
************************************************************************
DEADLINES:
jobnr.  deadline
  2       35
  4       34
  6       9
  7       26
************************************************************************
