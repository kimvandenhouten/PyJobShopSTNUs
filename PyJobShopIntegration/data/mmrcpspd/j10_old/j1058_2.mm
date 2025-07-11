************************************************************************
file with basedata            : mm58_.bas
initial value random generator: 367020778
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  71
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       14        2       14
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   8
   3        3          2           7  11
   4        3          2           6  10
   5        3          1           7
   6        3          2           9  11
   7        3          2           9  10
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
  2      1     4       5    5    0    4
         2     6       5    5    2    0
         3     9       4    4    0    4
  3      1     9       7   10    7    0
         2    10       4    3    7    0
         3    10       3    5    7    0
  4      1     1       8    6    0    5
         2     1       7    5    7    0
         3     3       3    3    7    0
  5      1     1      10    7    0    8
         2     2      10    6    0    5
         3     2      10    6    9    0
  6      1     2       6    4    0    6
         2     5       6    4    4    0
         3    10       5    3    3    0
  7      1     3       5    5    6    0
         2     5       5    3    5    0
         3     6       5    2    2    0
  8      1     6       5   10    0    7
         2     8       5   10    0    5
         3    10       5   10    6    0
  9      1     1       7    9    5    0
         2     1       7   10    0    5
         3     4       5    8    3    0
 10      1     2       9    3    9    0
         2     5       9    2    6    0
         3    10       6    1    0    4
 11      1     2       5   10    3    0
         2     4       5    6    2    0
         3     7       4    6    0    6
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   22   27   34   23
************************************************************************
DEADLINES:
jobnr.  deadline
  4       65
  9       69
  10      36
************************************************************************
