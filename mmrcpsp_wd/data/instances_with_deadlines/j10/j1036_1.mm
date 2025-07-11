************************************************************************
file with basedata            : mm36_.bas
initial value random generator: 23340
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
    1     10      0       26        8       26
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           5
   3        3          3           8   9  11
   4        3          2          10  11
   5        3          2           6  11
   6        3          2           7   9
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
  2      1     8       5    8    0    8
         2     8       5    8    6    0
         3     8       9    7    8    0
  3      1     2      10    5    3    0
         2     8       8    4    0    8
         3    10       5    4    0    5
  4      1     3       4    4    9    0
         2     7       3    4    5    0
         3     7       3    2    6    0
  5      1     1       5   10    5    0
         2     2       4    7    4    0
         3     3       4    6    4    0
  6      1     4       7    4    9    0
         2     6       7    2    7    0
         3     6       7    1    8    0
  7      1     5       5    8    3    0
         2     5       5    8    0    4
         3     8       5    8    0    3
  8      1     7       7    2    7    0
         2     8       5    1    4    0
         3     9       3    1    0    4
  9      1     7      10    9    0   10
         2     8       6    8    0    7
         3    10       5    7    0    6
 10      1     1       9    4    7    0
         2     8       8    4    0    8
         3     9       7    1    0    6
 11      1     4       3    8    6    0
         2     5       1    7    0    4
         3     5       2    8    3    0
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   10   10   57   46
************************************************************************
DEADLINES:
jobnr.  deadline
  3       11
  5       4
  6       5
  8       56
  10      19
  11      35
************************************************************************
