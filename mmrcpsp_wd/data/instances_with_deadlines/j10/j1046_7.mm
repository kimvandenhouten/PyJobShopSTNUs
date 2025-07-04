************************************************************************
file with basedata            : mm46_.bas
initial value random generator: 2041637870
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  64
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       17        5       17
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           8   9  11
   3        3          1          11
   4        3          3           5   6   9
   5        3          2           7   8
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
  2      1     2       7    9    7    6
         2     2       7    9    8    5
         3     6       4    3    7    3
  3      1     1       8    6    7    9
         2     3       8    6    7    7
         3     6       6    5    6    5
  4      1     5       9    6    6    9
         2     5       9    8    5   10
         3    10       9    2    4    5
  5      1     5       3    8    8   10
         2     7       3    6    6    8
         3     7       3    6    5    9
  6      1     1      10    9    4    8
         2     2      10    8    3    8
         3     3       9    8    3    7
  7      1     4       8    7   10    7
         2     8       8    6    8    5
         3     9       7    4    4    4
  8      1     3      10    8    2   10
         2     7       9    7    2    7
         3    10       6    7    1    7
  9      1     3       7    8   10    6
         2     3       7    9    7    6
         3     4       5    7    1    6
 10      1     3       7    4    8    6
         2     6       6    2    4    3
         3     6       7    4    6    1
 11      1     3       9    9    7    6
         2     3       9    9    5    7
         3     3       9   10    5    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   17   18   55   66
************************************************************************
DEADLINES:
jobnr.  deadline
  2       31
  3       4
  7       31
  8       55
************************************************************************
