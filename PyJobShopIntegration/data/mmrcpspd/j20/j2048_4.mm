************************************************************************
file with basedata            : md368_.bas
initial value random generator: 1060857520
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  157
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       16        4       16
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           8  14  17
   3        3          3          11  12  15
   4        3          3           5   7  18
   5        3          3           6   9  13
   6        3          2          14  19
   7        3          2          14  20
   8        3          2          10  13
   9        3          3          15  17  19
  10        3          2          12  15
  11        3          3          13  17  18
  12        3          1          16
  13        3          1          21
  14        3          1          21
  15        3          1          20
  16        3          2          18  19
  17        3          1          21
  18        3          1          20
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     5      10    4    8    7
         2     6      10    4    6    4
         3    10       9    4    3    2
  3      1     3       3    3    8    8
         2     4       3    3    3    8
         3     5       3    3    1    7
  4      1     2       7    9    9    6
         2     5       6    6    6    5
         3     8       4    3    2    5
  5      1     4       7    7    6    6
         2     5       7    6    6    4
         3     5       7    6    4    6
  6      1     3      10    8    9    5
         2     4      10    7    8    4
         3    10      10    5    8    4
  7      1     1       7    5    7    5
         2     1       7    5    6    6
         3     7       6    1    4    3
  8      1     2       4    4    4    8
         2     3       4    3    3    8
         3    10       1    3    2    7
  9      1     2       5    6    8    6
         2     2       4    6    9    6
         3     3       2    5    6    6
 10      1     2       7    3    3    7
         2     6       5    3    2    4
         3    10       4    2    1    1
 11      1     5       6    6    9    9
         2     6       5    5    7    6
         3     8       5    5    3    4
 12      1     1       6    4    2    7
         2     4       4    3    2    5
         3     5       4    3    1    5
 13      1     1       6    7    7    8
         2     5       5    4    7    6
         3     9       4    4    3    4
 14      1     4       7    7    5    4
         2     7       7    6    5    4
         3    10       4    6    3    4
 15      1     4      10    6    3    2
         2    10       4    4    3    2
         3    10       6    6    1    2
 16      1     1       8    3   10    9
         2     8       8    3    6    8
         3    10       8    3    5    7
 17      1     1       9    8    5    2
         2     8       7    8    4    2
         3     9       7    7    2    2
 18      1     3      10   10    8    7
         2     5       7    8    6    7
         3     5       8    8    6    5
 19      1     1       8    4    2    8
         2     2       8    4    2    5
         3     8       6    4    1    4
 20      1     2       6   10    8    8
         2     4       5    9    7    6
         3     5       3    9    6    4
 21      1     1       2    7    7    7
         2     5       2    6    6    3
         3    10       2    6    4    3
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   35   27   98  107
************************************************************************
DEADLINES:
jobnr.  deadline
  5       134
  7       121
  9       24
  10      5
  12      103
  13      35
  14      65
  15      121
  16      43
  17      65
  20      94
  21      5
************************************************************************
