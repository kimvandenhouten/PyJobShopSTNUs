************************************************************************
file with basedata            : mm34_.bas
initial value random generator: 25091
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  79
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       15        8       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           7
   3        3          2           5   7
   4        3          2          10  11
   5        3          1           6
   6        3          2           8  10
   7        3          2          10  11
   8        3          2           9  11
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       5    5    4    0
         2     4       4    5    3    0
         3     4       5    5    0    6
  3      1     5       5    3    0    9
         2     7       4    3   10    0
         3     8       4    2    0    9
  4      1     2      10    9    8    0
         2     8       6    7    7    0
         3     9       4    4    5    0
  5      1     2       8   10    0    8
         2     2       8   10    3    0
         3     9       8    9    0    8
  6      1     2       7    6    0    3
         2     2       7    5    3    0
         3     6       4    1    2    0
  7      1     5       9    8    0    6
         2     6       5    7    5    0
         3     9       4    6    0    6
  8      1     3       9    9    0    8
         2     4       7    4    0    6
         3    10       5    2    4    0
  9      1     1       4    3    9    0
         2     6       4    3    0    7
         3     8       3    3    7    0
 10      1     2       4    7   10    0
         2     5       4    5    6    0
         3     8       4    3    0    6
 11      1     3       4    5    9    0
         2     5       2    2    0    4
         3     8       2    1    0    4
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   10   11   35   29
************************************************************************
DEADLINES:
jobnr.  deadline
  2       72
  3       66
  4       7
  6       58
  8       41
  9       45
  10      37
  11      64
************************************************************************
