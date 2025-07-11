************************************************************************
file with basedata            : mb14_.bas
initial value random generator: 765458245
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  86
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
   3        3          1           5
   4        3          3           6   7   8
   5        3          1           6
   6        3          3           9  10  11
   7        3          2           9  11
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
  2      1     7       9    7    5    9
         2     8       9    6    3    9
         3     8       8    7    2    8
  3      1     1       7    9    7    5
         2     5       3    9    5    5
         3     6       1    7    5    4
  4      1     2      10    7   10    3
         2     4       7    3    8    2
         3     7       5    1    5    2
  5      1     2       6    7    3    6
         2     4       5    6    2    5
         3     9       2    5    2    5
  6      1     4       3    5   10    8
         2     4       3    9    8    8
         3    10       1    5    7    6
  7      1     3       9    8    2    9
         2     7       6    6    2    8
         3     9       5    5    2    8
  8      1     4       3    8    5    5
         2     9       2    8    3    5
         3    10       1    4    2    4
  9      1     1      10   10    9    7
         2     7       8    8    5    7
         3     9       7    6    5    2
 10      1     8       7    9    8    5
         2     8       7   10    7    5
         3     9       6    7    4    5
 11      1     4       7    6    7    7
         2     4       9    6    8    6
         3     9       6    4    5    4
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   12   11   67   64
************************************************************************
DEADLINES:
jobnr.  deadline
  2       52
  6       26
  11      72
************************************************************************
