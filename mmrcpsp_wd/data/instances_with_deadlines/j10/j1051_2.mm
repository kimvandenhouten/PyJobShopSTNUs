************************************************************************
file with basedata            : mm51_.bas
initial value random generator: 1399278337
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  77
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       14        6       14
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6   7   9
   3        3          1           8
   4        3          2           5  10
   5        3          2           6   9
   6        3          1           8
   7        3          2          10  11
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
  2      1     1       5    7    0    7
         2     1       4    8    0    7
         3     1       5    7    7    0
  3      1     4      10    9    6    0
         2     4      10    8    0    8
         3     7      10    5    7    0
  4      1     1       9    5    0    3
         2     2       8    4    0    2
         3     7       3    4    2    0
  5      1     1       9    4    0    2
         2     9       9    2    0    2
         3    10       8    2    3    0
  6      1     1       8    9    0    9
         2     4       7    5    6    0
         3     7       2    2    1    0
  7      1     1       8    9    0    2
         2     3       8    8    0    2
         3    10       7    7    2    0
  8      1     5       7    6    0    6
         2     6       5    5    9    0
         3     9       3    5    0    4
  9      1     1      10    6    0    7
         2     2       9    6    0    6
         3     9       9    5    3    0
 10      1     1       7    7    0    6
         2     6       7    7    8    0
         3     7       6    7    8    0
 11      1     5       5    5    0    7
         2     9       4    4    0    7
         3    10       1    3    3    0
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   28   24   35   40
************************************************************************
DEADLINES:
jobnr.  deadline
  2       62
  5       61
  7       20
  10      22
************************************************************************
