************************************************************************
file with basedata            : md340_.bas
initial value random generator: 28564
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  169
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       25        2       25
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6  10
   3        3          3           5  10  15
   4        3          1           9
   5        3          3           9  14  18
   6        3          3           7   8  16
   7        3          2          11  15
   8        3          3           9  12  18
   9        3          1          13
  10        3          1          17
  11        3          2          18  21
  12        3          3          13  14  15
  13        3          1          21
  14        3          1          17
  15        3          1          20
  16        3          3          19  20  21
  17        3          1          19
  18        3          2          19  20
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       4    0    3    0
         2     5       2    0    0    7
         3     6       0    4    3    0
  3      1     6       0    3   10    0
         2     7       6    0    0    4
         3     7       0    3    0    8
  4      1     4       0    7    0    6
         2     8       0    7    4    0
         3     9       0    6    3    0
  5      1     1       0    4    0    8
         2     3       4    0    0    7
         3    10       2    0    0    6
  6      1     2       9    0    8    0
         2    10       4    0    6    0
         3    10       5    0    5    0
  7      1     6       0    5    7    0
         2     7       4    0    0   10
         3     9       0    4    6    0
  8      1     5       6    0    0   10
         2     8       0    5    8    0
         3     8       0    5    0    3
  9      1     2      10    0    0    8
         2     7       8    0    0    7
         3     7       0    4    6    0
 10      1     5       7    0    4    0
         2     6       5    0    4    0
         3     6       3    0    0    4
 11      1     2       0    7    0    7
         2     9       0    6    0    6
         3    10       6    0    8    0
 12      1     7       0    9    0    5
         2     7       0    8    3    0
         3     8       1    0    0    5
 13      1     5       3    0    0    5
         2     8       0    7    5    0
         3    10       3    0    4    0
 14      1     2       0    4    0    4
         2     7       0    4    7    0
         3     9       0    3    0    2
 15      1     2       3    0    0    9
         2     6       0    9    0    6
         3    10       3    0    0    5
 16      1     5       0    5    7    0
         2     9       2    0    0    8
         3    10       0    3    6    0
 17      1     6       0    4    7    0
         2     7       3    0    4    0
         3    10       3    0    3    0
 18      1     3       0    8    0    9
         2     6       0    7    0    9
         3     7       0    6    7    0
 19      1     2       0    8    4    0
         2     5       0    8    3    0
         3     6       0    7    0    5
 20      1     2       4    0    6    0
         2     8       0    4    0    3
         3    10       2    0    5    0
 21      1     1       0    6    8    0
         2     3       0    4    8    0
         3     7       0    4    7    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   26   21   88   90
************************************************************************
DEADLINES:
jobnr.  deadline
  3       64
  8       31
  10      64
  11      161
  13      59
  14      64
  15      99
  18      108
************************************************************************
