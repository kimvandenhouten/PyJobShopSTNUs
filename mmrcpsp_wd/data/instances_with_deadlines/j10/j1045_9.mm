************************************************************************
file with basedata            : mm45_.bas
initial value random generator: 286010497
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  73
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       15        7       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6  10
   3        3          1           7
   4        3          2           8  11
   5        3          1           8
   6        3          2           7   8
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
  2      1     1       8    3    4    9
         2     7       8    2    2    9
         3     8       7    2    1    8
  3      1     6      10    9    8    5
         2     7       8    7    7    4
         3     8       8    6    7    3
  4      1     1       6    9    7    7
         2     6       6    9    5    6
         3    10       2    9    3    3
  5      1     1       7    7    9    7
         2     6       7    5    9    6
         3     8       7    5    8    5
  6      1     3       7    6    6    7
         2     6       7    5    4    6
         3     9       5    5    3    6
  7      1     5       5    2    7    6
         2     5       4    3    7    6
         3     9       2    1    3    6
  8      1     2       3    1    8    4
         2     2       3    4    3    4
         3     2       3    3    3    5
  9      1     1       5    9    8    5
         2     2       5    8    8    4
         3     3       4    8    7    4
 10      1     5       3    8    9    9
         2     7       3    8    7    4
         3     8       2    7    7    2
 11      1     4       9    8    5    8
         2     6       9    7    4    6
         3     8       6    7    1    2
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   18   20   49   48
************************************************************************
DEADLINES:
jobnr.  deadline
  2       11
  8       40
  11      38
************************************************************************
