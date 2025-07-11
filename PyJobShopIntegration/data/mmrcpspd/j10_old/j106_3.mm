************************************************************************
file with basedata            : mm6_.bas
initial value random generator: 864870188
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  89
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       22        0       22
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           5
   3        3          3           6   9  11
   4        3          3           5   8   9
   5        3          1           7
   6        3          1          10
   7        3          2          10  11
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
  2      1     3      10    0    7   10
         2     4       0    7    6    7
         3     7       0    4    5    7
  3      1     2       9    0    9    7
         2     4       0    9    6    6
         3     8       9    0    1    6
  4      1     8       0    2    5    7
         2     8       8    0    4    8
         3    10       6    0    2    6
  5      1     3       0    5    9    7
         2     6       0    4    9    6
         3     9       0    2    9    5
  6      1     2       7    0    8    9
         2     7       7    0    4    9
         3    10       6    0    4    6
  7      1     3       5    0    9    4
         2     9       0    7    7    4
         3    10       0    6    2    4
  8      1     9       0    4    9    8
         2     9       4    0    9    7
         3    10       2    0    7    5
  9      1     2       5    0   10    4
         2     2       6    0   10    3
         3     9       3    0   10    2
 10      1     5       8    0    7    6
         2     8       3    0    6    6
         3     9       0    7    3    1
 11      1     3       0    2    6    9
         2     3      10    0    9   10
         3     7       9    0    6    3
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   10    5   66   59
************************************************************************
DEADLINES:
jobnr.  deadline
  2       29
  11      26
************************************************************************
