************************************************************************
file with basedata            : mm15_.bas
initial value random generator: 2137802398
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  78
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       17        7       17
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           7
   3        3          2          10  11
   4        3          3           5   6  10
   5        3          1           9
   6        3          2           7   8
   7        3          1           9
   8        3          2           9  11
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     4       8    0    3    6
         2     4       0    3    2    7
         3     7       8    0    2    4
  3      1     6       7    0    3    7
         2     8       6    0    2    6
         3     8       7    0    1    5
  4      1     6       9    0    7    8
         2     7       8    0    3    8
         3    10       0    7    1    6
  5      1     4       0    9   10    9
         2     4       9    0    9    7
         3     7       0    6    9    7
  6      1     3      10    0    3    4
         2     7       0    8    3    1
         3     7       9    0    3    2
  7      1     3       0    9    8    6
         2     7       0    8    6    6
         3     8       0    8    5    6
  8      1     3       0    8    7    7
         2     3       2    0    8    7
         3    10       2    0    5    7
  9      1     5       8    0    9   10
         2     8       0    5    8    7
         3     8       6    0    6    6
 10      1     1       1    0    7    9
         2     1       0    8    6   10
         3     4       0    5    6    6
 11      1     5       0   10    2    7
         2     6       4    0    1    6
         3     9       3    0    1    4
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   17   54   68
************************************************************************
DEADLINES:
jobnr.  deadline
  6       11
************************************************************************
