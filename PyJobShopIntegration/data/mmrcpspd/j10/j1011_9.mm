************************************************************************
file with basedata            : mm11_.bas
initial value random generator: 2114911361
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  72
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       16        3       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           9
   3        3          2           7   8
   4        3          2           5  11
   5        3          1           6
   6        3          2           7   9
   7        3          1          10
   8        3          3           9  10  11
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       5    0    9    0
         2     3       4    0    8    0
         3     7       0    7    8    0
  3      1     1       0    7    0    7
         2     4       8    0    8    0
         3     4       0    7    0    5
  4      1     4      10    0    0    6
         2     5       0    8    0    4
         3     9       7    0    0    2
  5      1     5       4    0    0    7
         2     6       2    0    0    1
         3     8       0    9    4    0
  6      1     1       5    0    2    0
         2     2       4    0    2    0
         3     4       0    2    1    0
  7      1     5       0    7    0    3
         2     6       0    7    9    0
         3    10       4    0    0    2
  8      1     1       0    8    0    8
         2     3       0    7    6    0
         3     5       0    5    0    8
  9      1     4       0    2    9    0
         2     7       9    0    9    0
         3    10       4    0    8    0
 10      1     1       0    4    5    0
         2     4       9    0    3    0
         3     8       0    3    0    3
 11      1     3       0    6    0    7
         2     7       0    5    7    0
         3     7       6    0    6    0
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   12   14   46   29
************************************************************************
DEADLINES:
jobnr.  deadline
  3       71
  4       58
  5       9
  7       16
  8       14
  9       72
  10      1
  11      11
************************************************************************
