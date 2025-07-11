************************************************************************
file with basedata            : mm11_.bas
initial value random generator: 786786407
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  75
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       23        5       23
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           9  10
   3        3          2           7  11
   4        3          3           5   8   9
   5        3          1           6
   6        3          2          10  11
   7        3          1           8
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
  2      1     3       0    3    7    0
         2     5       9    0    0    8
         3     5       0    3    4    0
  3      1     1       0    7    0    7
         2     2       5    0    7    0
         3     7       5    0    0    7
  4      1     3       4    0    5    0
         2     5       0    8    0    2
         3    10       0    6    0    1
  5      1     8       0    5    7    0
         2     9       0    2    0    3
         3     9       0    1    4    0
  6      1     6       2    0    0    1
         2     7       0    1    5    0
         3     7       1    0    4    0
  7      1     3       5    0    0    8
         2     5       0    8    3    0
         3     9       0    7    0    5
  8      1     1       8    0    6    0
         2     3       0    2    5    0
         3     9       8    0    5    0
  9      1     5       0    3    8    0
         2     5      10    0    9    0
         3     8       7    0    7    0
 10      1     1       0    8    8    0
         2     2       0    8    0    6
         3     4       0    8    0    3
 11      1     6       2    0    3    0
         2     6       0    7    3    0
         3     7       0    6    3    0
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    9   14   47   25
************************************************************************
DEADLINES:
jobnr.  deadline
  8       68
  11      44
************************************************************************
