************************************************************************
file with basedata            : mm32_.bas
initial value random generator: 1806538605
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
    1     10      0       15        6       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           8
   3        3          1           8
   4        3          2           5   8
   5        3          3           6   7  11
   6        3          2           9  10
   7        3          2           9  10
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
  2      1     4       5    0    8    3
         2     4       0    8    8    4
         3     6       5    0    8    1
  3      1     2       0    9    5    4
         2     3       0    8    5    4
         3     4       0    6    2    3
  4      1     1       0    7    4    8
         2     4       2    0    4    7
         3     6       1    0    4    5
  5      1     6       0   10    5    5
         2     8       9    0    4    5
         3    10       0   10    4    4
  6      1     3       0   10    8    8
         2     8       0    7    5    7
         3    10       7    0    3    7
  7      1     4       7    0    7    7
         2     9       0    2    7    6
         3    10       7    0    6    5
  8      1     1       0    3    9    7
         2     4       5    0    5    6
         3     4       5    0    7    3
  9      1     1       8    0    6    7
         2    10       7    0    5    3
         3    10       0    5    5    3
 10      1     4       6    0    8    7
         2     6       0    7    7    5
         3     8       4    0    5    4
 11      1     2       2    0    8    4
         2     3       0    6    4    3
         3     6       1    0    3    3
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   27   68   61
************************************************************************
DEADLINES:
jobnr.  deadline
  2       64
  4       64
  5       29
  6       70
  7       21
  8       13
  9       30
  10      57
  11      40
************************************************************************
