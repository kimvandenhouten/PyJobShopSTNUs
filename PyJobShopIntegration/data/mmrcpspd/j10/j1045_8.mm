************************************************************************
file with basedata            : mm45_.bas
initial value random generator: 35099206
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  76
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
   2        3          2           5   7
   3        3          1           8
   4        3          1          10
   5        3          3           6   8  11
   6        3          1          10
   7        3          3           9  10  11
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
  2      1     5       6    6    8    7
         2     6       4    4    8    6
         3     6       6    6    7    7
  3      1     3       7    3    4    9
         2     7       5    3    4    8
         3     9       2    3    3    7
  4      1     3       5    6    6    5
         2     3       4    7    5    5
         3     5       4    5    2    5
  5      1     3       1    8    2    9
         2     5       1    7    1    9
         3    10       1    6    1    8
  6      1     1       8    4    9    4
         2     2       7    4    5    3
         3     8       5    3    4    3
  7      1     2       8    4    7    4
         2     2       7    4    9    3
         3     4       5    3    4    1
  8      1     1       7    8    9    9
         2     7       5    8    9    6
         3     8       3    8    9    2
  9      1     5      10    5    7    3
         2     5      10    5    9    2
         3     8      10    5    5    1
 10      1     3       4    6    7    7
         2     8       4    5    5    5
         3    10       3    2    2    3
 11      1     7       2    3   10    6
         2     7       2    2    9    9
         3     8       1    1    4    4
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   14   12   47   45
************************************************************************
DEADLINES:
jobnr.  deadline
  6       73
  7       62
************************************************************************
