************************************************************************
file with basedata            : mm32_.bas
initial value random generator: 133209496
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
    1     10      0       16        1       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   7
   3        3          3           6   7   9
   4        3          1           6
   5        3          3           8   9  11
   6        3          1          10
   7        3          1          11
   8        3          1          10
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     6       8    0    9    9
         2     8       0    6    8    9
         3     9       0    3    8    8
  3      1     2       0    5   10    9
         2     5       0    5    7    7
         3    10      10    0    7    4
  4      1     2       8    0    3    9
         2     6       5    0    3    9
         3     8       2    0    2    8
  5      1     3       9    0    7    8
         2     8       9    0    6    7
         3     9       8    0    5    7
  6      1     1       8    0    5    6
         2     6       4    0    5    6
         3     9       0    9    2    6
  7      1     4       0    4    9    6
         2     7       3    0    8    5
         3     7       3    0    9    4
  8      1     1       0    4   10    4
         2     8       9    0   10    4
         3    10       0    4   10    3
  9      1     7      10    0    9    6
         2     8       0    3    5    6
         3     9       6    0    4    3
 10      1     2       8    0    6    2
         2     5       3    0    4    1
         3     5       2    0    3    2
 11      1     1       0    3   10    9
         2     9       7    0    9    8
         3    10       5    0    9    8
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   30   15   78   68
************************************************************************
DEADLINES:
jobnr.  deadline
  2       9
  3       38
  4       81
  5       25
  6       73
  7       85
  8       4
  9       20
  10      59
  11      7
************************************************************************
