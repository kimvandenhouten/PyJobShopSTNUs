************************************************************************
file with basedata            : mm64_.bas
initial value random generator: 84032999
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  59
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
   2        3          1           6
   3        3          2           5  10
   4        3          2           7  11
   5        3          2           9  11
   6        3          2           7  11
   7        3          2           8   9
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
  2      1     1       4    6    7    8
         2     2       2    6    6    5
         3     2       3    6    4    5
  3      1     1       9    6    9   10
         2     2       8    5    9    7
         3     5       8    3    7    5
  4      1     6       3    9    7    8
         2     7       2    9    6    7
         3     9       2    8    4    7
  5      1     5       6    3    3    6
         2     7       4    3    1    6
         3     7       4    3    2    4
  6      1     1       6   10    9    8
         2     5       3    9    6    7
         3     5       4    9    6    5
  7      1     6       9    6   10    6
         2     6       8    9    8    4
         3     6       9    7    9    4
  8      1     2       3    8   10    7
         2     2       3    8    9    8
         3     5       3    8    8    4
  9      1     2       9    7    5    3
         2     4       9    6    4    2
         3     7       8    5    3    2
 10      1     2       3    7    9    6
         2     3       3    4    7    5
         3     5       2    2    6    5
 11      1     2       8   10    7    5
         2     8       8   10    5    3
         3     8       7   10    5    4
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   17   22   76   68
************************************************************************
DEADLINES:
jobnr.  deadline
  5       57
  6       8
  8       47
  9       16
  10      47
************************************************************************
