************************************************************************
file with basedata            : mm45_.bas
initial value random generator: 225922842
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
    1     10      0       10        8       10
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6   7   8
   3        3          2           9  11
   4        3          3           5   6   7
   5        3          1          11
   6        3          1          10
   7        3          1           9
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
  2      1     2       8    5    6    9
         2     5       7    4    5    8
         3     6       5    3    3    8
  3      1     5       3    4    9    8
         2     6       3    4    7    6
         3    10       1    3    6    4
  4      1     1       8    9    7    5
         2     2       5    8    7    5
         3     3       3    8    7    4
  5      1     3       2    9   10   10
         2     9       2    7    9    7
         3    10       2    6    9    4
  6      1     2       6    7    5    9
         2     8       6    6    4    9
         3    10       5    3    4    8
  7      1     5       8    5    4    8
         2     9       8    4    4    7
         3    10       7    4    2    5
  8      1     3       5    6    5    6
         2     3       4    6    5   10
         3     9       3    5    2    2
  9      1     3       4   10    7    3
         2     3       5    9    7    3
         3     8       3    9    5    2
 10      1     1       9    5    5    4
         2     2       9    5    5    2
         3     8       9    4    5    2
 11      1     1       7    7    9    3
         2     3       4    2    7    2
         3     3       2    2    5    3
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   17   20   52   47
************************************************************************
DEADLINES:
jobnr.  deadline
  2       5
  4       70
  6       60
  7       13
  8       28
  9       22
  10      39
  11      68
************************************************************************
