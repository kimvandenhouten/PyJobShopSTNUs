************************************************************************
file with basedata            : mm26_.bas
initial value random generator: 829674294
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
    1     10      0       17        2       17
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5  11
   3        3          2           8  11
   4        3          1           9
   5        3          2           6   7
   6        3          2           9  10
   7        3          2           8  10
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
  2      1     6       0    6    7    0
         2     7       0    6    5    0
         3    10       0    5    4    0
  3      1     4       3    0    9    0
         2     8       0    6    8    0
         3     9       3    0    5    0
  4      1     3       9    0    0   10
         2     5       0    5    5    0
         3    10       0    4    0    9
  5      1     2       0    4   10    0
         2     3       8    0    7    0
         3     3       9    0    0    2
  6      1     5       0    8    0    6
         2     6       8    0    9    0
         3     9       6    0    9    0
  7      1     4       0    4    0    7
         2     5       0    4    3    0
         3     7       9    0    0    6
  8      1     3       0    9    8    0
         2     5       4    0    0    9
         3    10       0    9    0    7
  9      1     2       7    0    0    9
         2     9       6    0    0    7
         3    10       1    0    3    0
 10      1     1       0    9    3    0
         2     1       4    0    0    5
         3     8       0   10    0    5
 11      1     6       7    0    0    2
         2     6       8    0    5    0
         3     7       6    0    0    2
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   25   19   36   25
************************************************************************
DEADLINES:
jobnr.  deadline
  2       20
  3       22
  4       19
  5       76
  6       40
  7       21
  8       10
  9       52
  10      52
  11      7
************************************************************************
