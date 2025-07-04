************************************************************************
file with basedata            : mm52_.bas
initial value random generator: 985058676
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  79
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       15        2       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   9
   3        3          2           7   8
   4        3          2           7   9
   5        3          2           6   7
   6        3          2           8  11
   7        3          1          11
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
  2      1     2      10    3    0    5
         2     2       9    3    9    0
         3     6       8    3    9    0
  3      1     5       3    4    4    0
         2     7       2    2    3    0
         3     8       1    2    0    9
  4      1     2       7    4    0    9
         2     2       7    6    9    0
         3     7       7    4    9    0
  5      1     3       7    6    7    0
         2     8       7    6    0    6
         3    10       5    5    0    5
  6      1     6       3    5    0    8
         2     9       3    4    0    8
         3    10       3    2    4    0
  7      1     3       8   10    7    0
         2     6       4   10    0    2
         3     7       1    9    5    0
  8      1     1       5    5    0    3
         2     5       5    4    9    0
         3     9       5    3    6    0
  9      1     1       9    2    0    8
         2     3       8    1    0    8
         3     9       8    1    0    7
 10      1     1       5   10    0    6
         2     3       5    7    4    0
         3     6       4    6    1    0
 11      1     4      10    9    0   10
         2     7       3    4    0   10
         3     7       5    4    0    9
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   16   53   66
************************************************************************
DEADLINES:
jobnr.  deadline
  2       34
  3       20
  4       3
  6       68
  7       51
  8       11
  9       18
************************************************************************
