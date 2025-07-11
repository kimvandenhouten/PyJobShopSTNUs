************************************************************************
file with basedata            : mm22_.bas
initial value random generator: 1898479828
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  82
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       12        1       12
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           7   8  11
   3        3          1           6
   4        3          2           5   6
   5        3          2           7  10
   6        3          1          10
   7        3          1           9
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
  2      1     4       0    6    8    6
         2     8       4    0    7    4
         3    10       0    4    6    3
  3      1     4       7    0    5    8
         2     5       0    8    5    7
         3     8       7    0    4    3
  4      1     3       5    0   10    8
         2     3       0    4   10    7
         3    10       7    0   10    7
  5      1     6       0    7   10    9
         2     9       0    7   10    5
         3    10       0    6   10    1
  6      1     1       0    8    9    4
         2     5       9    0    6    3
         3     8       6    0    6    3
  7      1     1       0    7   10    9
         2     2       0    4    5    6
         3     5       0    2    2    4
  8      1     3       0    6   10   10
         2     3       6    0   10   10
         3     9       0    6    8    7
  9      1     2       5    0    5    3
         2     3       4    0    4    3
         3     6       4    0    3    2
 10      1     1       9    0    8    4
         2     4       0    6    7    4
         3     8       8    0    5    3
 11      1     1       0    9   10    8
         2     6       0    7   10    7
         3     8       5    0    9    4
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   23   74   53
************************************************************************
DEADLINES:
jobnr.  deadline
  2       27
  3       10
  4       61
  5       70
  6       68
  7       27
  8       59
  10      82
  11      69
************************************************************************
