************************************************************************
file with basedata            : md370_.bas
initial value random generator: 1783822897
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  163
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       28       16       28
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   8
   3        3          3           6  10  19
   4        3          3          11  17  18
   5        3          3           7  10  12
   6        3          3           9  17  18
   7        3          1          19
   8        3          2           9  15
   9        3          1          20
  10        3          3          13  14  16
  11        3          3          13  16  20
  12        3          3          13  15  19
  13        3          1          21
  14        3          1          15
  15        3          2          17  18
  16        3          1          21
  17        3          1          20
  18        3          1          21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     5       0   10    5    7
         2     6       4    0    4    5
         3    10       0    7    2    4
  3      1     1       0    8    7    7
         2     4       0    7    6    5
         3     6       7    0    6    3
  4      1     5       7    0    5    5
         2     7       0    3    4    5
         3     9       0    2    1    3
  5      1     4       0    9    7    2
         2     5       9    0    4    2
         3     8       0    4    1    2
  6      1     5       2    0    6    2
         2     6       2    0    5    2
         3    10       0    4    5    1
  7      1     6       2    0    9    4
         2     8       0    4    9    3
         3     9       0    3    9    2
  8      1     3       8    0    7    6
         2     4       5    0    6    5
         3     5       5    0    4    2
  9      1     2       5    0    7    3
         2     4       0    7    5    3
         3    10       3    0    1    2
 10      1     1       6    0    7    8
         2     6       0    9    7    7
         3    10       0    8    7    4
 11      1     1       0    7   10    2
         2     3       0    7    8    2
         3     5      10    0    8    1
 12      1     1       8    0    3    7
         2    10       0    1    3    7
         3    10       0    4    3    6
 13      1     1       0    5    1    3
         2     2       0    5    1    2
         3     3       4    0    1    2
 14      1     6       0    6    7    8
         2     7       8    0    6    6
         3     7       0    6    7    6
 15      1     3       7    0    9    6
         2     8       7    0    9    3
         3    10       0    6    7    3
 16      1     3       0    2    6    4
         2     4       0    1    5    3
         3     6       9    0    4    3
 17      1     1       0    5    6    6
         2     4       0    4    5    6
         3    10       0    2    4    5
 18      1     4       9    0    5    5
         2     7       8    0    2    2
         3     7       0    6    2    5
 19      1     3       0    9    5    8
         2    10       0    8    4    8
         3    10      10    0    5    7
 20      1     5       2    0    4    7
         2     8       2    0    4    6
         3     9       0    5    4    2
 21      1     5       4    0    8    5
         2     9       0    5    2    5
         3     9       3    0    5    5
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   17   13  113   95
************************************************************************
DEADLINES:
jobnr.  deadline
  2       23
  3       112
  6       119
  7       137
  10      75
  13      29
  14      122
  15      7
  17      28
  19      118
  20      28
  21      8
************************************************************************
