************************************************************************
file with basedata            : mm13_.bas
initial value random generator: 407312940
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  80
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       10        2       10
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           5
   3        3          2           6   7
   4        3          3           6   7  10
   5        3          1           9
   6        3          2           8   9
   7        3          2           8   9
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
  2      1     2       8    0    9    5
         2     5       8    0    8    4
         3     7       0    2    7    4
  3      1     2       0    9    8    4
         2     6       0    9    5    2
         3    10       0    8    3    2
  4      1     1       0    8    2    6
         2     9       0    6    2    6
         3     9       3    0    1    6
  5      1     3       4    0    4    4
         2     6       0    6    3    4
         3     7       0    3    3    3
  6      1     2       4    0    9   10
         2     6       3    0    4   10
         3    10       0    3    2   10
  7      1     1       2    0    6    8
         2     1       0    7    5    5
         3     6       0    6    4    5
  8      1     5       0    8    2    6
         2     5       6    0    2    5
         3     7       4    0    1    2
  9      1     1       5    0    5    9
         2     7       0    8    3    7
         3     9       0    4    2    4
 10      1     1       8    0    7    9
         2     8       0    7    7    8
         3     8       4    0    6    9
 11      1     1       4    0    6    9
         2     2       4    0    4    9
         3     7       4    0    3    7
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    9   14   37   55
************************************************************************
DEADLINES:
jobnr.  deadline
  11      42
************************************************************************
