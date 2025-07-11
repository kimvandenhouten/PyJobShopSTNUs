************************************************************************
file with basedata            : mm63_.bas
initial value random generator: 1181582883
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
    1     10      0       10        0       10
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           7  11
   3        3          1           8
   4        3          3           5   7   8
   5        3          1           6
   6        3          1          11
   7        3          1           9
   8        3          3           9  10  11
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     4       5    6    9   10
         2     7       5    4    9    4
         3     8       4    3    9    3
  3      1     1       6    9    7    7
         2     4       5    5    4    4
         3     4       6    4    6    5
  4      1     1       8    5    4   10
         2     2       7    3    4    8
         3     6       4    3    2    7
  5      1     4       5    5    6    5
         2     5       5    3    6    5
         3     8       5    2    6    3
  6      1     2       5    6    7    9
         2     9       4    6    5    7
         3    10       4    5    5    6
  7      1     2      10    9    6    8
         2     4      10    6    5    8
         3     8      10    6    5    5
  8      1     2       7    4    6    4
         2    10       7    1    6    3
         3    10       7    3    5    3
  9      1     3       8    6    3    9
         2     8       6    5    3    5
         3    10       4    5    3    3
 10      1     7      10    6    9    6
         2     9      10    1    6    3
         3     9       9    3    5    4
 11      1     2       8    3    8    2
         2     5       8    3    7    1
         3     9       7    2    3    1
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   26   21   60   60
************************************************************************
DEADLINES:
jobnr.  deadline
  2       59
  4       62
  7       47
  9       69
************************************************************************
