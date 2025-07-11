************************************************************************
file with basedata            : mm46_.bas
initial value random generator: 1039523274
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  70
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
   2        3          2           6   7
   3        3          1           5
   4        3          3           6   8  11
   5        3          2           6  11
   6        3          1          10
   7        3          2           9  11
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
  2      1     1       4    4    9   10
         2     4       4    3    9    6
         3     9       3    2    9    5
  3      1     2       7    9    7    4
         2     6       6    7    4    4
         3     7       3    4    4    4
  4      1     1       2    4    6    4
         2     3       2    3    5    3
         3     8       1    3    5    3
  5      1     1       7    7    5    9
         2     5       6    6    5    8
         3     6       3    5    5    8
  6      1     1       3    6    7    7
         2     3       2    5    6    6
         3     4       2    4    5    6
  7      1     5       5    8    4    7
         2     5       8    8    4    5
         3     7       2    7    4    4
  8      1     3      10    7    3   10
         2     7      10    5    3   10
         3     7       9    6    1   10
  9      1     4      10    8    9    7
         2     4      10    9    6    6
         3     9      10    8    3    4
 10      1     2       5    9    5    5
         2     2       4    8    6    5
         3     4       3    5    3    5
 11      1     1       8    6    7    4
         2     1       7    6   10    4
         3     9       7    6    2    4
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   18   16   54   60
************************************************************************
DEADLINES:
jobnr.  deadline
  3       14
  4       59
  5       3
  6       27
  7       52
  8       20
  11      24
************************************************************************
