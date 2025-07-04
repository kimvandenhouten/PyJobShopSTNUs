************************************************************************
file with basedata            : mm21_.bas
initial value random generator: 1869837808
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  73
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       11        3       11
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           6
   3        3          3           5   7   8
   4        3          2           5   9
   5        3          1          11
   6        3          2           7  11
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
  2      1     2       9    0    9    8
         2     5       0    2    6    5
         3     7       8    0    3    3
  3      1     3       0    3    3    9
         2     4      10    0    2    8
         3     6       7    0    2    7
  4      1     2       0    2    9    7
         2     5       2    0    9    6
         3     7       2    0    7    5
  5      1     1       5    0    7    5
         2     4       4    0    4    5
         3     9       0    4    4    4
  6      1     5       9    0    5    2
         2     8       0    5    4    2
         3     8       8    0    5    1
  7      1     1       0    8    5   10
         2     2       9    0    4   10
         3     6       0    6    4    9
  8      1     3       0    3    7    8
         2     4       9    0    7    7
         3     8       8    0    6    4
  9      1     1       0    3    5    8
         2     9       8    0    5    6
         3    10       6    0    4    3
 10      1     3       7    0    5    9
         2     4       0    4    4    5
         3     5       6    0    3    2
 11      1     1       3    0    9    8
         2     2       0   10    5    7
         3     7       0    7    2    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   16   13   44   50
************************************************************************
DEADLINES:
jobnr.  deadline
  9       48
  11      46
************************************************************************
