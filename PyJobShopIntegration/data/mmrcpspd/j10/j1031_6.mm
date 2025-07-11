************************************************************************
file with basedata            : mm31_.bas
initial value random generator: 667009696
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  84
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       20        2       20
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6   8
   3        3          2           5   6
   4        3          1           8
   5        3          2           7  10
   6        3          1          11
   7        3          2           9  11
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
  2      1     2       8    0   10    7
         2     4       7    0   10    6
         3     5       4    0    9    3
  3      1     5       5    0    8    6
         2     9       3    0    6    6
         3    10       2    0    6    5
  4      1     7       7    0    4    8
         2     9       0    9    4    5
         3    10       0    7    2    5
  5      1     1       0   10    2    9
         2     6       0   10    2    7
         3    10       0   10    2    4
  6      1     5       0   10    9    7
         2     5       6    0    8    7
         3     8       0    9    7    6
  7      1     3       5    0   10    8
         2     3       0    7   10    8
         3     6       5    0    6    3
  8      1     6       9    0    2    8
         2     8       0    6    2    8
         3     9       7    0    2    6
  9      1     3       6    0   10    7
         2     4       0    7    7    6
         3     9       6    0    4    5
 10      1     7       0    7    8   10
         2     8       9    0    7    9
         3    10       0    5    5    7
 11      1     5       0    9    6    7
         2     5       2    0    5    9
         3     7       1    0    2    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   21   29   62   70
************************************************************************
DEADLINES:
jobnr.  deadline
  2       71
  3       16
  4       24
  5       77
  6       71
  7       35
  8       76
  9       20
************************************************************************
