************************************************************************
file with basedata            : md360_.bas
initial value random generator: 39119824
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  174
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       27        4       27
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   8  17
   3        3          3          14  17  18
   4        3          3           6  11  13
   5        3          1           7
   6        3          3           9  15  16
   7        3          3           9  13  15
   8        3          3          15  18  20
   9        3          3          10  12  19
  10        3          2          14  21
  11        3          3          16  18  19
  12        3          1          20
  13        3          1          14
  14        3          1          20
  15        3          1          19
  16        3          1          17
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
  2      1     3       9    3    5    6
         2     6       8    3    3    4
         3     9       7    2    2    1
  3      1     1       5    9    8    3
         2     1       7    7    7    4
         3     9       3    5    6    2
  4      1     4       7    7    7    4
         2     7       6    5    7    4
         3     8       4    4    5    4
  5      1     6       6    5   10    8
         2     7       6    4   10    8
         3     9       6    3   10    5
  6      1     1       7    5    9    4
         2     9       6    4    6    3
         3    10       5    3    4    2
  7      1     7      10    6    4   10
         2     7       9    9    4   10
         3     8       9    4    4   10
  8      1     2       5    9   10    9
         2     5       5    7   10    7
         3    10       5    7   10    5
  9      1     1       5   10    6   10
         2     2       2    8    4    5
         3     5       2    4    3    3
 10      1     3       8    8    6    4
         2     5       7    6    5    2
         3     8       7    3    3    1
 11      1     7       6    4    9    9
         2    10       4    2    7    9
         3    10       4    3    4    9
 12      1     2       9   10    6    9
         2     4       6    7    4    8
         3    10       5    3    3    8
 13      1     3       4    6    4    7
         2     6       4    5    4    4
         3    10       3    4    3    1
 14      1     4       8    8    6    7
         2     4       9    7    4    7
         3    10       8    5    3    7
 15      1     8       8    8    9    6
         2     9       8    4    1    5
         3     9       7    4    5    3
 16      1     6       6    4   10    5
         2     8       4    3   10    4
         3    10       4    3   10    2
 17      1     7       7    8    7    9
         2     9       5    7    7    7
         3     9       6    7    6    8
 18      1     3       6    4   10    8
         2     3       7    5    8    8
         3     5       3    3    5    8
 19      1     2       9    4    7    6
         2     7       8    4    7    5
         3    10       6    3    5    5
 20      1     1       6    4    8    5
         2     4       3    3    8    4
         3     5       2    2    7    3
 21      1     3       9    7    6    9
         2     8       9    5    4    7
         3    10       8    5    1    3
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   36   40  108  102
************************************************************************
DEADLINES:
jobnr.  deadline
  4       97
  5       53
  6       148
  7       166
  9       45
  10      55
  11      139
  12      102
  13      135
  15      88
  16      103
  17      65
  18      152
  19      126
  20      57
************************************************************************
