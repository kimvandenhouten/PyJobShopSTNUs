************************************************************************
file with basedata            : mm64_.bas
initial value random generator: 16627
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
    1     10      0       16        4       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           6  10
   3        3          2           5   8
   4        3          2           7   8
   5        3          2           9  11
   6        3          2           9  11
   7        3          1           9
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
  2      1     2       5   10   10    8
         2     5       4    9   10    8
         3     7       3    9   10    5
  3      1     4       6    7    6    2
         2     5       4    7    5    2
         3     9       3    6    5    1
  4      1     3       5    5    4    8
         2     4       3    3    3    7
         3    10       2    2    3    5
  5      1     4       5    4    5    3
         2     6       4    4    3    3
         3    10       3    1    3    2
  6      1     3       3    6    7    7
         2     3       4    7    7    6
         3     4       2    4    5    5
  7      1     7       6    5    8    5
         2     8       5    5    6    4
         3     9       5    5    5    3
  8      1     1      10    7    5    9
         2     2       9    7    4    7
         3     9       7    6    2    5
  9      1     6       7    7    6    8
         2     9       6    6    4    3
         3     9       4    6    3    6
 10      1     5       8    6    7   10
         2     6       7    4    6    6
         3     9       5    1    6    2
 11      1     1      10    9   10    5
         2     6       8    7   10    5
         3     8       6    1   10    5
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   25   23   68   65
************************************************************************
DEADLINES:
jobnr.  deadline
  2       8
  3       14
  5       64
  6       79
  10      2
  11      2
************************************************************************
