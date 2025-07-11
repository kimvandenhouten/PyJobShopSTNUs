************************************************************************
file with basedata            : mm34_.bas
initial value random generator: 393728935
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
    1     10      0       15        7       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           8  10
   3        3          2           8   9
   4        3          2           5   7
   5        3          3           6   9  10
   6        3          1           8
   7        3          1          11
   8        3          1          11
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     5       8   10    0    7
         2     5       8    9    4    0
         3    10       6    8    1    0
  3      1     3       4    8    3    0
         2     8       4    7    0    9
         3    10       2    7    0    5
  4      1     3      10    7    0    6
         2     7       8    6    2    0
         3    10       6    3    1    0
  5      1     2       7    6    0    9
         2     7       7    6    0    8
         3    10       4    5    0    7
  6      1     2       5    9    7    0
         2     5       4    9    0    3
         3     7       4    9    6    0
  7      1     7       8   10    8    0
         2     8       5    6    0    8
         3    10       3    3    4    0
  8      1     5       2    9    0    6
         2     5       2    9   10    0
         3     7       1    9    9    0
  9      1     1       4    8    0    3
         2     3       2    5    5    0
         3     6       2    5    4    0
 10      1     4       7    9    0    9
         2     8       6    4    0    8
         3     8       7    6    4    0
 11      1     3       6    2    0    7
         2     6       4    1    0    4
         3     6       5    2   10    0
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   10   14   27   37
************************************************************************
DEADLINES:
jobnr.  deadline
  8       33
************************************************************************
