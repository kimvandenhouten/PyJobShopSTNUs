************************************************************************
file with basedata            : mm59_.bas
initial value random generator: 2128067762
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
    1     10      0       15        9       15
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           7
   3        3          2           5   6
   4        3          2           7  10
   5        3          1          11
   6        3          3           9  10  11
   7        3          2           8  11
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
  2      1     2       9    7    0    9
         2     3       8    7    0    9
         3     8       5    6    0    9
  3      1     5       4    4    0    9
         2     5       4    4    8    0
         3     9       2    4    0    9
  4      1     8       7    4    0    5
         2    10       4    3    7    0
         3    10       4    3    0    5
  5      1     4       3   10    9    0
         2     9       3   10    0    9
         3    10       3    9    8    0
  6      1     1       9    5    9    0
         2     5       6    4    7    0
         3    10       4    1    0    5
  7      1     1       3   10    0    3
         2     4       2   10   10    0
         3     7       1   10   10    0
  8      1     5       6    8    0    6
         2     6       5    8    7    0
         3     9       5    4    7    0
  9      1     1       7    5   10    0
         2     4       6    5    0    9
         3    10       5    5    6    0
 10      1     4       7   10    5    0
         2     5       7    9    4    0
         3     7       7    9    0    6
 11      1     3      10    5    0    7
         2     3       6    6    0    7
         3     8       2    4    0    5
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   23   30   46   52
************************************************************************
DEADLINES:
jobnr.  deadline
  2       54
  3       41
  4       76
  6       30
  7       86
  8       79
  9       55
  10      5
  11      83
************************************************************************
