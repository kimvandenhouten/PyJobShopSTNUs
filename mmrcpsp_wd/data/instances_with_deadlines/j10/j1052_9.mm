************************************************************************
file with basedata            : mm52_.bas
initial value random generator: 983424876
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  77
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       16        2       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   7
   3        3          2           5   6
   4        3          2           9  10
   5        3          1          11
   6        3          3           7   8  10
   7        3          1          11
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
  2      1     1       5    4    7    0
         2     3       4    4    5    0
         3     7       4    4    1    0
  3      1     3       8    7    0    8
         2     5       7    5    2    0
         3     6       6    5    0    7
  4      1     9       6    2    8    0
         2     9       8    2    0    7
         3    10       2    2    0    6
  5      1     2       2    6   10    0
         2     5       1    4    0    6
         3     7       1    4    9    0
  6      1     4       8    7    0   10
         2     5       7    7    0    6
         3     8       6    6    7    0
  7      1     4       8    8    0    4
         2     5       6    7    6    0
         3    10       5    4    6    0
  8      1     6       8    8    0    6
         2     7       6    8    1    0
         3     8       4    8    0    5
  9      1     2       8    7    0    5
         2     2       9    7    0    3
         3     4       6    5    6    0
 10      1     4       8    4    0    5
         2     5       5    3    2    0
         3    10       1    1    2    0
 11      1     5       3    4    0    4
         2     6       3    2    0    4
         3     7       2    2    0    4
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   19   16   49   55
************************************************************************
DEADLINES:
jobnr.  deadline
  2       54
  3       2
  4       18
  5       28
  7       13
  8       8
  10      56
  11      28
************************************************************************
