************************************************************************
file with basedata            : mm39_.bas
initial value random generator: 790927754
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
    1     10      0       10        3       10
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           8
   3        3          3           6   9  10
   4        3          2           5   7
   5        3          3           6   8   9
   6        3          1          11
   7        3          1           8
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
  2      1     6       4    8    9    9
         2     8       4    8    8    9
         3     9       3    5    7    8
  3      1     6       7    8    9    8
         2     8       7    4    8    8
         3     9       6    3    4    6
  4      1     2       6    8    7    9
         2     3       4    6    5    5
         3    10       4    5    1    2
  5      1     2       4    8    7   10
         2     8       4    8    7    8
         3     9       3    5    6    6
  6      1     1       5    5    9    7
         2     3       5    3    9    7
         3     6       5    2    8    6
  7      1     2       8    5    9   10
         2    10       6    5    7    5
         3    10       4    4    8    4
  8      1     3       7    4    9    5
         2     7       5    4    7    4
         3     9       5    4    6    4
  9      1     3       3    8    7    8
         2     6       3    8    5    8
         3     8       1    8    2    7
 10      1     1       6    5    4    1
         2     3       5    3    2    1
         3    10       5    3    1    1
 11      1     1       7    7    6    9
         2     2       6    4    5    8
         3     6       3    2    3    8
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    9   12   67   69
************************************************************************
DEADLINES:
jobnr.  deadline
  2       66
  3       39
  4       42
  5       38
  6       60
  7       12
  8       46
  9       45
  10      24
  11      35
************************************************************************
