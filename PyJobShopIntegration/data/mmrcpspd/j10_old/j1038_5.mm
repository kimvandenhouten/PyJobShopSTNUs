************************************************************************
file with basedata            : mm38_.bas
initial value random generator: 24237588
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
    1     10      0       16        4       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2          10  11
   3        3          2          10  11
   4        3          1           5
   5        3          3           6   7  11
   6        3          1           8
   7        3          2           9  10
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
  2      1     4       5    7    6    2
         2     4       7    5    6    2
         3     5       3    4    5    1
  3      1     1       8    6    6    5
         2     6       6    5    5    3
         3     8       4    5    4    3
  4      1     1       7    8   10    7
         2     3       6    5   10    6
         3     5       3    1    9    1
  5      1     1       6    7    7    6
         2     1       5    8    5    6
         3     7       3    7    2    5
  6      1     5       3    5    8    6
         2     5       3    6    9    5
         3     7       3    4    6    4
  7      1     1       7    3    5    3
         2     4       7    2    4    3
         3     8       3    2    4    1
  8      1     1       8    2   10    6
         2     4       8    1    6    6
         3     6       7    1    4    6
  9      1     8       7    7    6    8
         2    10       5    4    6    5
         3    10       5    5    6    4
 10      1     1       9    8    3    7
         2     2       9    6    3    6
         3     8       8    4    2    5
 11      1     4       4    5    8    6
         2     5       2    3    4    6
         3     8       2    2    3    5
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   11   10   58   46
************************************************************************
DEADLINES:
jobnr.  deadline
  4       49
  5       72
  6       15
  7       65
  8       60
  9       30
  10      1
  11      70
************************************************************************
