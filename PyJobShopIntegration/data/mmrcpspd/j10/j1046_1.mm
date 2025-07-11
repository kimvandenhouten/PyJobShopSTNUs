************************************************************************
file with basedata            : mm46_.bas
initial value random generator: 24785
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
    1     10      0       14        6       14
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           5
   3        3          2           6   9
   4        3          2           7  11
   5        3          2           8  10
   6        3          2          10  11
   7        3          2           9  10
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
  2      1     4       8    4    7    6
         2     4      10    5    7    5
         3     7       5    4    6    2
  3      1     5       7    9    6    6
         2     8       4    9    5    4
         3     9       2    9    5    2
  4      1     5       2   10   10    7
         2     7       2    9    6    6
         3     8       2    9    6    4
  5      1     4       6    6    8    2
         2     7       5    5    7    1
         3     7       5    4    5    2
  6      1     3      10    8    8    9
         2     9       9    7    7    8
         3    10       9    6    6    6
  7      1     1       9   10    7    9
         2     4       5    7    3    9
         3     6       4    3    2    6
  8      1     1      10    3   10    7
         2     5       8    2    9    7
         3     9       7    2    9    5
  9      1     3       8    3    6    3
         2     7       8    3    5    3
         3    10       7    3    4    3
 10      1     6       6    9    6    7
         2     9       5    9    4    7
         3    10       3    8    4    5
 11      1     6       8    8    3    7
         2     9       5    7    2    6
         3    10       2    3    1    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   17   17   60   52
************************************************************************
DEADLINES:
jobnr.  deadline
  3       75
************************************************************************
