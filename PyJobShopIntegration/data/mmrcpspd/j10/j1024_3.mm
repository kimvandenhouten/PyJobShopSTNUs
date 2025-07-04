************************************************************************
file with basedata            : mm24_.bas
initial value random generator: 330259129
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  76
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       10        5       10
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   6
   3        3          1          11
   4        3          3           6   9  11
   5        3          2           7   8
   6        3          1          10
   7        3          2           9  11
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
  2      1     2       0   10   10    2
         2     4       4    0    5    1
         3     7       0    9    5    1
  3      1     2       0    2    6   10
         2     5       7    0    6    6
         3     7       5    0    5    4
  4      1     1       8    0    9    4
         2     3       8    0    8    4
         3     5       7    0    8    3
  5      1     2       0    8    8    7
         2     3       0    8    4    6
         3     9       0    6    2    6
  6      1     2       0    4    4    6
         2     3       0    3    3    6
         3     4       0    3    2    4
  7      1     3       0    2    5    5
         2    10       0    1    2    4
         3    10       7    0    1    4
  8      1     1       8    0    7    4
         2     8       4    0    6    4
         3    10       0    9    6    3
  9      1     3       0    8    6    8
         2     5       5    0    6    6
         3     7       0    7    5    3
 10      1     2       3    0    7    6
         2     2       2    0    5    7
         3     8       2    0    1    4
 11      1     1       4    0   10    5
         2     4       0    6    9    4
         3     9       0    5    9    3
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   12   72   58
************************************************************************
DEADLINES:
jobnr.  deadline
  2       68
  3       72
  5       70
  8       52
  9       55
  10      48
************************************************************************
