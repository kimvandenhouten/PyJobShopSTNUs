************************************************************************
file with basedata            : mm39_.bas
initial value random generator: 1046882282
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
    1     10      0       17        9       17
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   6
   3        3          3           5   7   8
   4        3          2          10  11
   5        3          1          11
   6        3          1           7
   7        3          2           9  10
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
  2      1     5       5    6    9    6
         2     9       5    6    9    5
         3    10       4    5    7    4
  3      1     9       7    8    5    7
         2     9       7    9    4    8
         3    10       5    5    4    6
  4      1     2       6    3    8    8
         2     8       4    2    7    1
         3     8       3    2    6    4
  5      1     2       4    6    7    6
         2     4       3    6    5    6
         3     7       3    6    4    6
  6      1     2       6   10    6    9
         2     3       4    5    4    7
         3     9       4    3    3    5
  7      1     1       3    6    8    3
         2     5       3    4    5    3
         3     7       2    3    4    2
  8      1     2       6    9    8    9
         2     5       4    9    8    8
         3    10       4    7    7    6
  9      1     7       8    4    4    8
         2     7      10    6    7    7
         3     7       9    6    8    7
 10      1     4       8    7    8    8
         2     4       7   10    7    9
         3     9       4    7    3    8
 11      1     1       6   10    5    4
         2     4       6    6    3    4
         3     7       6    4    2    3
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   12   12   64   63
************************************************************************
DEADLINES:
jobnr.  deadline
  3       56
  4       31
  5       7
  7       17
  9       28
  11      40
************************************************************************
