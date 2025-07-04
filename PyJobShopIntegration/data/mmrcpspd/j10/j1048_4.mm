************************************************************************
file with basedata            : mm48_.bas
initial value random generator: 1054231221
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  86
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
   2        3          3           5   6   7
   3        3          1           6
   4        3          2           5   7
   5        3          1           9
   6        3          1          10
   7        3          2           8  11
   8        3          2           9  10
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     4       8    5    6    7
         2     4       7    5    6    9
         3    10       7    5    6    3
  3      1     2       8    7    6    9
         2     3       8    6    6    9
         3     5       7    5    5    9
  4      1     3       9    9    9    4
         2     6       7    8    7    2
         3     7       7    8    6    1
  5      1     2       6    5    8   10
         2     7       2    5    6    8
         3     7       6    5    7    5
  6      1     2       7    8    6    3
         2     6       7    8    3    3
         3     9       6    7    3    3
  7      1     1       5    9    7    7
         2     6       4    8    4    5
         3    10       4    7    2    4
  8      1     1       9    1    4    5
         2     3       7    1    4    5
         3    10       6    1    2    5
  9      1     2      10    7    8    4
         2     9       9    5    8    3
         3    10       9    5    8    2
 10      1     1       3    3    2    8
         2    10       1    2    2    3
         3    10       2    3    1    4
 11      1     2      10    6    6   10
         2     6       7    5    5    9
         3     8       4    5    2    7
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   21   15   62   69
************************************************************************
DEADLINES:
jobnr.  deadline
  5       82
  6       73
  7       25
************************************************************************
