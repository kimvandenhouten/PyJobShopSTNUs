************************************************************************
file with basedata            : mm2_.bas
initial value random generator: 589997716
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
    1     10      0       11        2       11
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1          11
   3        3          3           5   7   8
   4        3          3           6   9  11
   5        3          2           9  11
   6        3          1          10
   7        3          1           9
   8        3          1          10
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     5       0    6    0    7
         2     8       3    0    8    0
         3    10       0    6    0    5
  3      1     2       0    7    0    3
         2     3       0    5    1    0
         3     3       1    0    1    0
  4      1     3       0    8    7    0
         2     3       8    0   10    0
         3     8       0    7    0    1
  5      1     4       5    0   10    0
         2     5       5    0    0    2
         3     8       4    0    0    2
  6      1     3       0    3    0    9
         2     8       2    0    5    0
         3     8       2    0    0    8
  7      1     3       8    0    0    1
         2     3       8    0    8    0
         3     4       0    9    5    0
  8      1     6       9    0    8    0
         2     7       0    5    0    8
         3     9       0    2    6    0
  9      1     1       0    8    6    0
         2     2       5    0    0    8
         3     8       0    4    6    0
 10      1     3       0    5    0    2
         2     3       0    5    9    0
         3     8       0    4    0    3
 11      1     1       0    9    0    9
         2    10       4    0    3    0
         3    10       0    7    0    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    9    9   34   26
************************************************************************
DEADLINES:
jobnr.  deadline
  2       69
  3       64
  4       33
  5       26
  6       34
  7       21
  9       73
  10      56
  11      68
************************************************************************
