************************************************************************
file with basedata            : mm29_.bas
initial value random generator: 265748271
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  81
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       14        9       14
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5  11
   3        3          1           6
   4        3          2           7   9
   5        3          2           8  10
   6        3          3           7   9  11
   7        3          1          10
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
  2      1     1       0    9    6    6
         2     4       0    7    4    5
         3     8       0    5    1    4
  3      1     6       7    0    7    6
         2     9       0    4    6    5
         3    10       0    4    2    4
  4      1     5       2    0   10    2
         2     6       2    0    9    1
         3     7       0    9    6    1
  5      1     1       0    7    9    9
         2     9       0    6    8    6
         3     9       2    0    8    7
  6      1     2       0   10    7    5
         2     6       0    8    6    4
         3     8       8    0    4    3
  7      1     2       0    8    7    3
         2     7       2    0    6    3
         3     9       0    7    2    2
  8      1     3       9    0   10    7
         2     3       0    8    7    7
         3     5       0    8    5    7
  9      1     1       7    0    8    4
         2     6       0    7    7    3
         3     9       7    0    5    1
 10      1     1       0    4    5    7
         2     8       0    3    5    3
         3     8       8    0    5    1
 11      1     6       5    0    8   10
         2     8       0    9    5    5
         3     8       4    0    6    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   17   24   50   39
************************************************************************
DEADLINES:
jobnr.  deadline
  2       79
  3       31
  4       41
  5       26
  7       18
  8       62
  9       69
  11      77
************************************************************************
