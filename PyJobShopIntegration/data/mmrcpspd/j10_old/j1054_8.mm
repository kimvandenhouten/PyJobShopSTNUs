************************************************************************
file with basedata            : mm54_.bas
initial value random generator: 507943858
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
    1     10      0        8        4        8
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           6  10
   3        3          3           5   9  11
   4        3          2           9  10
   5        3          1           7
   6        3          2           8  11
   7        3          1          10
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
  2      1     2       8    9    9    2
         2     4       7    8    9    1
         3     7       5    8    7    1
  3      1     3       7    3    6    7
         2     8       6    3    4    6
         3    10       4    3    2    6
  4      1     5       4    6    8    7
         2     5       7    4    8    7
         3     8       4    2    8    7
  5      1     1       9   10    7    4
         2     1       9    6    9    5
         3     6       7    6    3    4
  6      1     2       1    9    8    8
         2     8       1    7    5    8
         3     9       1    3    2    8
  7      1     3       4    6    7   10
         2     6       3    5    6   10
         3     8       2    3    3   10
  8      1     2       4    2    9    8
         2     3       2    2    9    4
         3     9       1    1    8    3
  9      1     2       6    2    8    8
         2     5       3    2    5    7
         3     9       3    2    1    6
 10      1     1       3    9    5    7
         2     7       3    7    5    7
         3     9       3    6    3    7
 11      1     3       6    8    5    5
         2     5       4    8    5    5
         3     9       3    8    4    5
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   18   20   58   62
************************************************************************
DEADLINES:
jobnr.  deadline
  9       43
  10      80
************************************************************************
