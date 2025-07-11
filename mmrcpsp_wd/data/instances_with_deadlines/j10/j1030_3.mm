************************************************************************
file with basedata            : mm30_.bas
initial value random generator: 1222280666
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  69
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       17        4       17
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           9
   3        3          2          10  11
   4        3          3           5   6   7
   5        3          1          11
   6        3          2           8  11
   7        3          2           8   9
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
  2      1     3       0    4    3    3
         2     4       9    0    2    2
         3     4       0    3    3    1
  3      1     2       0   10    7    2
         2     3      10    0    7    1
         3     7       0    8    7    1
  4      1     1       9    0    4    9
         2     2       0   10    4    5
         3     2       9    0    4    5
  5      1     3       4    0    7    8
         2     6       0    3    7    7
         3    10       0    3    6    2
  6      1     3       0    5    5    6
         2     5       8    0    2    6
         3     5       7    0    3    6
  7      1     8       0    2    7    8
         2     8      10    0    6    8
         3    10       8    0    6    5
  8      1     5       9    0    8   10
         2     6       5    0    8    8
         3     9       0    7    8    5
  9      1     1       0    7   10    7
         2     6       0    4    3    2
         3     6       7    0    2    5
 10      1     3       7    0    7    8
         2     7       0   10    6    7
         3     8       7    0    6    7
 11      1     5       0    7    6    8
         2     5       4    0    6    8
         3     8       0    7    4    7
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   41   24   56   55
************************************************************************
DEADLINES:
jobnr.  deadline
  3       65
  6       50
  11      19
************************************************************************
