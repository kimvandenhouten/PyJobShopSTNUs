************************************************************************
file with basedata            : mm15_.bas
initial value random generator: 922411328
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
    1     10      0       10        6       10
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           6   9
   3        3          2           5   7
   4        3          1          10
   5        3          2          10  11
   6        3          3           7   8  10
   7        3          1          11
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
  2      1     1       0    8    3    3
         2     5       0    6    3    3
         3     9       7    0    3    2
  3      1     4       2    0    7    4
         2     5       2    0    7    3
         3     6       1    0    6    3
  4      1     2       7    0    9    5
         2     7       0    7    8    5
         3     8       7    0    8    4
  5      1     1       6    0    3    5
         2     2       0    3    2    4
         3     7       4    0    2    4
  6      1     2       0    8    6    8
         2     5       7    0    5    4
         3     8       0    6    5    4
  7      1     3       0   10    4    8
         2     5       6    0    3    8
         3     5       0    9    3    8
  8      1     2       6    0    7    9
         2     5       0    9    7    9
         3    10       0    7    6    8
  9      1     5       4    0    7    5
         2     6       0    7    6    5
         3     9       0    3    6    4
 10      1     4       2    0   10    3
         2     5       0    6   10    2
         3     6       2    0   10    2
 11      1     3       6    0    4    6
         2     8       5    0    4    6
         3     9       4    0    4    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   10   18   58   53
************************************************************************
DEADLINES:
jobnr.  deadline
  2       41
  4       64
  5       16
  6       6
  9       57
  10      8
  11      50
************************************************************************
