************************************************************************
file with basedata            : mm21_.bas
initial value random generator: 47193057
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  82
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       12        0       12
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           5
   3        3          2           6  10
   4        3          3           8   9  10
   5        3          1           7
   6        3          1          11
   7        3          3           9  10  11
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
  2      1     1       0    6   10    6
         2     2       0    5    7    6
         3     3       0    4    6    5
  3      1     5       9    0    9    4
         2     6       6    0    9    3
         3    10       5    0    9    1
  4      1     2       9    0   10    8
         2     5       7    0   10    7
         3    10       0    5    9    7
  5      1     4       0    3    6    6
         2     6       5    0    6    4
         3     8       0    2    6    1
  6      1     3       6    0    7    6
         2     6       0    6    6    6
         3     9       5    0    3    5
  7      1     4       9    0    3    3
         2     7       9    0    2    2
         3     9       8    0    2    2
  8      1     1       7    0    7    2
         2     6       5    0    4    2
         3     6       5    0    5    1
  9      1     2       7    0    5    8
         2     6       0    9    5    8
         3    10       0    7    4    7
 10      1     3       8    0    7    8
         2     7       1    0    7    6
         3     8       0    2    7    4
 11      1     1       0    8    6    4
         2     3       2    0    4    3
         3     9       0    5    4    1
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   19   15   57   38
************************************************************************
DEADLINES:
jobnr.  deadline
  2       59
  3       14
  4       34
  5       42
  6       78
  7       31
  9       74
  10      67
  11      40
************************************************************************
