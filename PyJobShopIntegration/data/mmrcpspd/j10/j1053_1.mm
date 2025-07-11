************************************************************************
file with basedata            : mm53_.bas
initial value random generator: 9424
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  91
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       16        8       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   8
   3        3          1           6
   4        3          3           6   7   8
   5        3          2           9  11
   6        3          1          11
   7        3          2           9  10
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
  2      1     6       4    3   10    5
         2     7       4    3    6    4
         3     7       4    1    4    5
  3      1     1      10    8    7    7
         2     9      10    4    7    7
         3    10      10    2    6    7
  4      1     6       7    7    3    2
         2     9       5    5    3    2
         3    10       3    2    2    1
  5      1     2       6    7    8    6
         2     4       5    5    6    5
         3    10       3    3    1    4
  6      1     3       5    7    5    6
         2     5       4    6    5    6
         3     8       4    5    5    3
  7      1     1      10   10    5    8
         2     4       9    9    4    7
         3     6       8    9    3    5
  8      1     4       5    5    5    9
         2     4       6    5    4    9
         3    10       2    4    3    7
  9      1     3       2   10    2    8
         2     5       2   10    2    7
         3    10       1   10    1    4
 10      1     1       6   10    4    3
         2     6       5   10    4    3
         3    10       5    9    2    1
 11      1     7       9    9    5    9
         2    10       8    9    1    9
         3    10       9    9    3    8
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   22   23   33   48
************************************************************************
DEADLINES:
jobnr.  deadline
  2       88
  3       47
  4       71
  5       52
  7       91
  8       57
  9       54
  10      20
  11      91
************************************************************************
