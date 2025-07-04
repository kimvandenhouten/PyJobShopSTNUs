************************************************************************
file with basedata            : md354_.bas
initial value random generator: 976530614
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
    1     20      0       28        4       28
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5  10  16
   3        3          1           6
   4        3          3          12  15  17
   5        3          1          18
   6        3          3           7   8   9
   7        3          3          10  14  20
   8        3          1          11
   9        3          2          15  16
  10        3          2          13  17
  11        3          3          13  14  19
  12        3          3          13  14  16
  13        3          1          21
  14        3          1          18
  15        3          3          18  19  20
  16        3          2          19  20
  17        3          1          21
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
  2      1     3       5    0    5   10
         2     6       3    0    4    7
         3     8       0    2    4    7
  3      1     4       0    3    6    5
         2     7      10    0    4    5
         3     8       0    3    3    3
  4      1     1       7    0   10    5
         2     2       0    7    8    4
         3     6       6    0    7    3
  5      1     3      10    0    6    6
         2     5       8    0    4    3
         3    10       0    4    4    2
  6      1     2       0    7    5    8
         2     5       0    7    4    6
         3     8       8    0    4    4
  7      1     4       0    6    9    3
         2     8       6    0    1    3
         3     8       0    5    1    3
  8      1     1       6    0    8    8
         2     7       0   10    7    6
         3     8       4    0    6    4
  9      1     2       7    0    7   10
         2     3       0    7    6    7
         3     4       0    4    6    7
 10      1     3       4    0    7    7
         2     5       4    0    6    4
         3     7       3    0    5    1
 11      1     1       7    0    7    8
         2     9       7    0    4    7
         3    10       0    6    3    5
 12      1     3       0    6    9    7
         2     5       0    4    8    6
         3    10       3    0    7    6
 13      1     6       0    8   10    7
         2     8       4    0    6    6
         3     9       0    6    4    5
 14      1     7       9    0    5    6
         2     7       0    5    5    6
         3     8       0    4    4    4
 15      1     2       8    0    9    8
         2     8       0    8    9    5
         3    10       7    0    9    5
 16      1     5       0    9    8    6
         2     5       4    0    9    6
         3    10       0    9    8    5
 17      1     6       0    4    6    1
         2     7       8    0    4    1
         3    10       8    0    2    1
 18      1     6       5    0    6    9
         2     7       0    7    6    4
         3     8       0    6    6    4
 19      1     1       0    7   10    9
         2     4       0    7    9    8
         3     7       6    0    9    5
 20      1     3       9    0    9   10
         2     5       7    0    6    9
         3    10       5    0    4    9
 21      1     5       0    8    8    5
         2     7       0    3    5    4
         3    10       5    0    5    3
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   17   19  114   99
************************************************************************
DEADLINES:
jobnr.  deadline
  2       107
  3       92
  4       163
  5       60
  6       35
  7       38
  8       36
  9       126
  10      56
  11      10
  12      146
  13      156
  14      31
  15      2
  16      97
  17      119
  18      37
  19      34
  20      45
  21      104
************************************************************************
