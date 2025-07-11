************************************************************************
file with basedata            : mm29_.bas
initial value random generator: 339155955
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  83
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       16        6       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6   9
   3        3          3           7   9  11
   4        3          2           8   9
   5        3          1          10
   6        3          1           7
   7        3          1           8
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
  2      1     3       7    0    8    9
         2     4       0    7    8    7
         3     8       0    7    6    6
  3      1     2       7    0   10    8
         2     6       3    0    6    6
         3     8       0    7    3    4
  4      1     4      10    0   10    6
         2     7       0    2    5    5
         3    10      10    0    3    2
  5      1     2       6    0    9    7
         2     4       4    0    9    4
         3     9       0    6    7    2
  6      1     8       4    0    7    7
         2     9       0    3    5    5
         3    10       0    3    3    4
  7      1     2       4    0   10    5
         2     2       0    4    8    5
         3     7       4    0    6    5
  8      1     1       9    0    5    8
         2     5       8    0    3    7
         3     8       8    0    3    4
  9      1     5       4    0    4   10
         2     7       4    0    3    9
         3     8       3    0    3    9
 10      1     2       0    5   10    5
         2     7       0    4    7    5
         3     8       4    0    6    5
 11      1     1       0    3    5    7
         2     5       0    2    5    7
         3     7       0    2    4    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   24   18   51   52
************************************************************************
DEADLINES:
jobnr.  deadline
  8       56
  9       31
************************************************************************
