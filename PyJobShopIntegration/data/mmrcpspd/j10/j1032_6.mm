************************************************************************
file with basedata            : mm32_.bas
initial value random generator: 654816664
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  88
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       17        3       17
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           8   9  10
   3        3          1          11
   4        3          2           5   7
   5        3          3           6   8   9
   6        3          1          11
   7        3          1          10
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
  2      1     3       8    0    3    8
         2     4       0    8    3    8
         3    10       5    0    2    7
  3      1     1       0    2    5    5
         2     9       7    0    5    4
         3    10       5    0    4    1
  4      1     6       7    0    8    9
         2     7       0    3    6    6
         3     8       2    0    6    3
  5      1     1       0    8    8   10
         2     5       1    0    6    8
         3    10       0    7    5    7
  6      1     4       7    0   10    5
         2     5       7    0    8    5
         3     6       0    9    6    4
  7      1     1       0    7    7    8
         2     5       0    5    6    8
         3     9       9    0    1    5
  8      1     6       7    0    4   10
         2     7       0    5    4    9
         3     8       3    0    4    9
  9      1     1       0    7    8    4
         2     6       6    0    6    4
         3    10       0    7    5    3
 10      1     6       9    0    9    9
         2     7       0    5    5    8
         3    10       9    0    2    8
 11      1     4       0   10    2    5
         2     7       0    4    1    4
         3     7       1    0    1    4
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   29   26   64   73
************************************************************************
DEADLINES:
jobnr.  deadline
  4       69
  5       24
  7       57
  8       49
  9       81
************************************************************************
