************************************************************************
file with basedata            : mm53_.bas
initial value random generator: 904985334
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  88
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       16        8       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           7
   3        3          3           5   9  11
   4        3          3           6   7  11
   5        3          2           6   7
   6        3          1          10
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
  2      1     6       8    8    4    9
         2     7       8    5    2    9
         3     8       8    3    1    8
  3      1     3       3   10    5    9
         2    10       3   10    1    9
         3    10       3   10    3    8
  4      1     4       3   10    6    6
         2     6       3    7    3    4
         3     7       3    4    1    2
  5      1     1       8    8    5    5
         2     5       6    7    4    4
         3     7       5    6    2    4
  6      1     1       6   10    1    8
         2     8       5    5    1    7
         3    10       4    3    1    5
  7      1     4       9    7    7    5
         2     6       8    6    4    4
         3    10       8    5    3    4
  8      1     2       9    9    8    6
         2     6       9    8    7    6
         3     8       9    7    6    4
  9      1     3       9    3    2    8
         2     7       7    2    1    6
         3     8       6    1    1    3
 10      1     4       2    7    1    7
         2    10       1    2    1    2
         3    10       2    3    1    1
 11      1     1       6    6    7   10
         2    10       4    5    5    9
         3    10       3    5    6    9
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   23   23   27   53
************************************************************************
DEADLINES:
jobnr.  deadline
  2       23
  3       65
  5       71
  6       59
  8       66
  11      53
************************************************************************
