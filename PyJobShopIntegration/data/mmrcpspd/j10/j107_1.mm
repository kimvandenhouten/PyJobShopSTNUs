************************************************************************
file with basedata            : mm7_.bas
initial value random generator: 1823
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
    1     10      0       11        4       11
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           5
   3        3          3           6   7  10
   4        3          1           6
   5        3          2           9  10
   6        3          2           9  11
   7        3          1           8
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
  2      1     4       7    0    5   10
         2     6       0    1    4    8
         3    10       4    0    3    8
  3      1     1       0    7    9    7
         2     9       3    0    4    6
         3     9       2    0    5    6
  4      1     1       0    7    6    8
         2     6       0    5    5    8
         3     7       0    4    3    6
  5      1     1      10    0    8    8
         2     9       9    0    6    8
         3    10       8    0    4    8
  6      1     2       6    0    9    9
         2     6       4    0    9    9
         3     6       0   10    9    8
  7      1     1       0    9    7   10
         2     3       1    0    6    8
         3     4       0    3    4    5
  8      1     5       0    2    4    9
         2     5       8    0    3    6
         3     6       7    0    3    6
  9      1     4       9    0    8    9
         2     4       0    8    8    8
         3    10       9    0    5    8
 10      1     1       3    0    2   10
         2     2       0    7    2    9
         3     4       0    3    1    7
 11      1     4       0    8    5    7
         2     6       9    0    3    7
         3     8       7    0    2    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   10    8   56   81
************************************************************************
DEADLINES:
jobnr.  deadline
  2       16
  5       50
  7       28
  8       58
  9       58
************************************************************************
