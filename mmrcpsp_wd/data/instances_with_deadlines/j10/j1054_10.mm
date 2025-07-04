************************************************************************
file with basedata            : mm54_.bas
initial value random generator: 360407948
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
    1     10      0       18        6       18
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6   7   8
   3        3          3           5   6  10
   4        3          1          10
   5        3          1           8
   6        3          2           9  11
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
  2      1     6       7   10    7    9
         2     7       7    8    7    7
         3     8       3    4    7    7
  3      1     3      10    7    5    8
         2     5       5    7    4    6
         3     9       1    7    4    6
  4      1     4       9    5    8    7
         2     5       6    5    8    7
         3     8       5    2    7    6
  5      1     1       7    4    8    3
         2     4       5    4    8    2
         3     7       3    2    7    2
  6      1     5       7    7    7    5
         2     7       6    6    6    4
         3     9       5    3    6    4
  7      1     4       8    8    3    9
         2     4       9    7    3    9
         3     6       7    4    3    7
  8      1     5       7    7    5    8
         2     9       7    5    4    7
         3    10       6    4    4    6
  9      1     1       6    6    3    9
         2     1       5    6    4   10
         3    10       2    4    1    5
 10      1     5       7    2    8    2
         2     7       6    2    8    2
         3     9       6    1    7    2
 11      1     7      10    9    5    6
         2     7       8    9    6    6
         3    10       8    8    4    5
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   23   19   56   59
************************************************************************
DEADLINES:
jobnr.  deadline
  2       49
  3       83
  8       25
  10      15
  11      57
************************************************************************
