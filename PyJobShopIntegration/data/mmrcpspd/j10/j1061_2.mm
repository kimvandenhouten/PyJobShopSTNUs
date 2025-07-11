************************************************************************
file with basedata            : ma14_.bas
initial value random generator: 1009676374
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  12
horizon                       :  82
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     10      0       14        8       14
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           7   9
   3        3          3           5   8   9
   4        3          1           6
   5        3          1           7
   6        3          2          10  11
   7        3          1          10
   8        3          2          10  11
   9        3          1          12
  10        3          1          12
  11        3          1          12
  12        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     8       6   10    4   10
         2    10       6    9    3    9
         3    10       6   10    2   10
  3      1     2       5    8    4    9
         2     2       6    8    5    7
         3     6       2    8    4    3
  4      1     6       4    6    5    3
         2     6       3    8    5    3
         3     8       2    4    3    3
  5      1     1       6    5    5    7
         2     9       4    5    4    6
         3    10       3    4    4    2
  6      1     3       9    9    6    7
         2     6       7    9    5    5
         3     9       4    9    5    2
  7      1     3       9    7    4    8
         2     8       8    4    4    7
         3     9       8    1    4    7
  8      1     1       6    5    3    7
         2    10       6    2    3    4
         3    10       6    2    2    7
  9      1     3       9    3    2    5
         2     3       8    3    2    7
         3     8       7    2    2    5
 10      1     3       9    8    6    8
         2     3       6    8    5    9
         3     4       3    8    4    6
 11      1     2       4    8    4    6
         2     8       1    7    3    1
         3     8       1    6    4    3
 12      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   27   28   35   48
************************************************************************
DEADLINES:
jobnr.  deadline
  3       26
  8       50
************************************************************************
