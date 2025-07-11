************************************************************************
file with basedata            : mm53_.bas
initial value random generator: 638097787
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  67
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       11        9       11
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6   8
   3        3          2           5   9
   4        3          2           7  11
   5        3          1          10
   6        3          2           9  10
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
  2      1     3       6    7   10    8
         2     5       6    7   10    7
         3     6       4    6   10    7
  3      1     3       7    2    6    3
         2     5       7    1    5    2
         3     6       6    1    4    2
  4      1     2       6    8   10    9
         2     6       5    7    9    9
         3     8       4    6    9    7
  5      1     5       4    5    9    7
         2     9       3    5    7    7
         3    10       2    5    4    6
  6      1     3       4   10    6    3
         2     3       4    9    7    3
         3     9       4    7    5    2
  7      1     1       4    7    6    8
         2     6       4    6    4    7
         3     8       4    1    2    5
  8      1     2       7    8   10    9
         2     4       7    8    8    8
         3     5       6    5    6    8
  9      1     2       4    9    5    8
         2     2       3   10    5    8
         3     4       2    6    5    7
 10      1     3       9    6    4    5
         2     3       9    5    7    8
         3     3       7    5    4    9
 11      1     1       5    6    4    6
         2     2       3    4    3    6
         3     8       2    2    3    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   18   18   56   58
************************************************************************
DEADLINES:
jobnr.  deadline
  5       16
  6       38
  7       24
  8       23
************************************************************************
