************************************************************************
file with basedata            : mm60_.bas
initial value random generator: 1717789062
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
    1     10      0       13        6       13
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6   8
   3        3          1           9
   4        3          3           7   8   9
   5        3          1          10
   6        3          1           7
   7        3          1          11
   8        3          2          10  11
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       6    7    6    0
         2     9       5    7    0    7
         3    10       5    6    0    3
  3      1     4       2   10    0    4
         2     5       1    9    0    2
         3     6       1    9    0    1
  4      1     1       4    6    0    7
         2     3       4    5    0    2
         3    10       3    5    6    0
  5      1     4      10    5   10    0
         2     5      10    4    2    0
         3     5       9    5    0    8
  6      1     2       9    3    4    0
         2     6       7    2    4    0
         3     6       8    1    0    4
  7      1     6       8    7    0    7
         2     9       8    6    3    0
         3    10       7    6    0    1
  8      1     5      10    3    0    7
         2     7       9    3    2    0
         3     9       9    2    0    7
  9      1     2       3    8    0    9
         2     2       3    5    3    0
         3     5       2    1    2    0
 10      1     5       8   10    5    0
         2     6       6    5    2    0
         3     6       6    7    0    2
 11      1     2      10    8    7    0
         2     5      10    6    0    9
         3     7      10    6    4    0
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   32   23   46   64
************************************************************************
DEADLINES:
jobnr.  deadline
  2       10
  3       52
  4       13
  5       51
  6       23
  7       48
  8       21
  9       47
  10      7
  11      70
************************************************************************
