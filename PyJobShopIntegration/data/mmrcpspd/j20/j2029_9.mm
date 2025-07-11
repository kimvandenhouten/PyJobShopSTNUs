************************************************************************
file with basedata            : md349_.bas
initial value random generator: 774049507
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  160
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
   2        3          2          13  15
   3        3          3           5  10  11
   4        3          3           6   8  15
   5        3          3           6   7  13
   6        3          1           9
   7        3          2           9  12
   8        3          3          16  17  20
   9        3          2          17  18
  10        3          2          12  14
  11        3          1          21
  12        3          2          15  19
  13        3          2          17  20
  14        3          3          16  19  20
  15        3          1          16
  16        3          1          18
  17        3          2          19  21
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
  2      1     1       7    9    8    0
         2     3       7    8    7    0
         3     5       7    7    0    7
  3      1     2      10    7    0    3
         2     5       9    4    8    0
         3     8       9    4    5    0
  4      1     3       3    4    6    0
         2     6       3    2    5    0
         3     9       2    1    0    7
  5      1     5       7    8    9    0
         2     8       5    7    0    4
         3    10       3    4    0    3
  6      1     1       9    1   10    0
         2     5       6    1    6    0
         3     6       5    1    3    0
  7      1     1       8    5    4    0
         2     4       6    4    0    4
         3    10       4    2    3    0
  8      1     4      10    8    0    2
         2     5      10    7    4    0
         3     9       9    7    2    0
  9      1     6       5    9    9    0
         2     8       4    7    0   10
         3    10       4    5    0    5
 10      1     7       9    7    3    0
         2     7       6    9    2    0
         3    10       5    4    2    0
 11      1     6       3    8    0    8
         2     7       2    8    0    6
         3     9       2    6    0    5
 12      1     4       5   10    2    0
         2     4       5    9    0    5
         3     8       5    8    0    2
 13      1     2       2    6    7    0
         2     2       2    5    9    0
         3     3       1    5    4    0
 14      1     1       9    5    8    0
         2     4       7    4    4    0
         3     5       7    4    3    0
 15      1     2       7    7    7    0
         2     9       7    6    6    0
         3    10       6    6    2    0
 16      1     4       7    7    0    4
         2     5       5    7    0    3
         3     9       5    6    0    3
 17      1     4       9   10    0    3
         2     5       8    8    6    0
         3     6       8    4    0    2
 18      1     3       4    8    9    0
         2     5       4    8    0    8
         3    10       3    6    9    0
 19      1     4      10   10    7    0
         2     9       8    9    0    2
         3    10       4    9    0    2
 20      1     2       7    9    0    7
         2     2       7    9    8    0
         3     8       7    9    6    0
 21      1     3       8    7    2    0
         2     3       6    8    0    3
         3     5       5    7    0    1
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   15   15  119   77
************************************************************************
DEADLINES:
jobnr.  deadline
  3       62
  8       127
  11      14
  15      157
  16      81
************************************************************************
