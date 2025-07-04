************************************************************************
file with basedata            : md337_.bas
initial value random generator: 2071813968
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  162
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       22       16       22
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5  14  19
   3        3          1          21
   4        3          3           5   7   8
   5        3          2           6  16
   6        3          3          13  17  18
   7        3          3          10  15  19
   8        3          2           9  19
   9        3          3          12  13  15
  10        3          2          11  13
  11        3          2          12  17
  12        3          1          16
  13        3          2          20  21
  14        3          2          15  17
  15        3          1          16
  16        3          1          18
  17        3          1          20
  18        3          2          20  21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       0   10    9    0
         2     7       5    0    7    0
         3     8       0    8    3    0
  3      1     5       4    0    6    0
         2     9       0    2    0    7
         3    10       2    0    5    0
  4      1     5       0    1    0    8
         2     6       4    0    0    7
         3     8       3    0    0    6
  5      1     4       0    4    0    6
         2     4       0    7    9    0
         3     4       5    0    0    8
  6      1     1       7    0    0    4
         2     2       5    0    0    2
         3     6       0    3    0    1
  7      1     1       4    0   10    0
         2     3       2    0    8    0
         3     9       0    6    6    0
  8      1     1       0    9    5    0
         2     3       0    6    0    6
         3     5       0    5    3    0
  9      1     4       4    0    0    6
         2     5       3    0    0    6
         3     7       0    5    0    6
 10      1     1       8    0    8    0
         2     3       7    0    0    2
         3     9       0    7    5    0
 11      1     6       8    0    0    8
         2     6       8    0    3    0
         3     8       0    7    0    8
 12      1     1       0    2    0    7
         2     4       0    2    6    0
         3     6       0    1    5    0
 13      1     7       0    7    0    9
         2     7       0    4    8    0
         3    10       1    0    7    0
 14      1     6       0    3    8    0
         2     8      10    0    5    0
         3    10       0    3    0    4
 15      1     6       0    5    0   10
         2     8       6    0    6    0
         3     9       0    5    6    0
 16      1     1       0    8    0    7
         2     8       6    0    5    0
         3     9       6    0    3    0
 17      1     2       2    0    0    3
         2     5       0    7    4    0
         3    10       0    5    3    0
 18      1     2       7    0    0    6
         2     6       0    5    5    0
         3    10       0    3    0    5
 19      1     1       0    8    0    2
         2     6       0    7    4    0
         3    10       0    2    0    1
 20      1     2       0    3    0    6
         2     3       3    0    0    3
         3     7       0    2    0    2
 21      1     3      10    0    0    3
         2     4       0    5    0    3
         3     7       9    0    0    2
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
    7   11   74   84
************************************************************************
DEADLINES:
jobnr.  deadline
  2       24
  4       80
  5       105
  7       38
  8       120
  9       14
  10      115
  11      52
  13      66
  14      39
  15      162
  17      57
  19      146
  20      98
************************************************************************
